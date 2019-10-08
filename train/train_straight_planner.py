#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Train local planner using TRPO or CPO so that a vehicle can follow a
straight line at a specified target velocity.
"""

import argparse

import joblib
import lasagne.init as LI
import lasagne.layers as L
import lasagne.nonlinearities as LN
import numpy as np
import theano
#import theano.misc.pycuda_utils
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0')
from rllab.algos.trpo import TRPO
from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.envs.base import Env
from rllab.misc import ext, logger
from rllab.misc.instrument import run_experiment_lite, VariantGenerator
from rllab.misc.resolve import load_class
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.cpo.algos.safe.cpo import CPO
from sandbox.cpo.baselines.linear_feature_baseline import LinearFeatureBaseline
#import sandbox.rocky.tf.core.layers as T
from aa_simulation.envs.straight.straight_env import StraightEnv
from aa_simulation.safety_constraints.straight import StraightSafetyConstraint
#from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
#from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp


# Pre-trained policy and baseline
policy = None
baseline = None


def run_task(vv, log_dir=None, exp_name=None):
    global policy
    global baseline

    trpo_stepsize = 0.01
    trpo_subsample_factor = 0.2

    # Check if variant is available
    if vv['model_type'] not in ['BrushTireModel', 'LinearTireModel']:
        raise ValueError('Unrecognized model type for simulating robot')
    if vv['robot_type'] not in ['MRZR', 'RCCar']:
        raise ValueError('Unrecognized robot type')

    # Load environment
    if not vv['use_ros']:
        env = StraightEnv(
            target_velocity=vv['target_velocity'],
            dt=vv['dt'],
            model_type=vv['model_type'],
            robot_type=vv['robot_type'],
            mu_s=vv['mu_s'],
            mu_k=vv['mu_k']
        )
    else:
        from aa_simulation.envs.straight.straight_env_ros import StraightEnvROS
        env = StraightEnvROS(
            target_velocity=vv['target_velocity'],
            dt=vv['dt'],
            model_type=vv['model_type'],
            robot_type=vv['robot_type']
        )

    # Save variant information for comparison plots
    variant_file = logger.get_snapshot_dir() + '/variant.json'
    logger.log_variant(variant_file, vv)

    # Set variance for each action component separately for exploration
    # Note: We set the variance manually because we are not scaling our
    #       action space during training.
    init_std_speed = vv['target_velocity'] / 4
    init_std_steer = np.pi / 6
    init_std = [init_std_speed, init_std_steer]

    # Build policy and baseline networks
    # Note: Mean of policy network set to analytically computed values for
    #       faster training (rough estimates for RL to fine-tune).
    if policy is None or baseline is None:
        target_velocity = vv['target_velocity']
        target_steering = 0
        output_mean = np.array([target_velocity, target_steering])
        hidden_sizes = (32, 32)

        # In mean network, allow output b values to dominate final output
        # value by constraining the magnitude of the output W matrix. This is
        # to allow faster learning. These numbers are arbitrarily chosen.
        W_gain = min(vv['target_velocity'] / 5, np.pi / 15)


        # policy = GaussianLSTMPolicy(
        #     name="policy",
        #     env_spec=env.spec,
        #     lstm_layer_cls=T.TfBasicLSTMLayer,
        #     # gru_layer_cls=L.GRULayer,
        # )
        mean_network = MLP(
            input_shape=(env.spec.observation_space.flat_dim,),
            output_dim=env.spec.action_space.flat_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=LN.rectify,
            output_nonlinearity=None,
            output_W_init=LI.GlorotUniform(gain=W_gain),
            output_b_init=output_mean
        )
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            init_std=init_std,
            mean_network=mean_network
        )
        baseline = LinearFeatureBaseline(
            env_spec=env.spec,
            target_key='returns'
        )

    # Reset variance to re-enable exploration when using pre-trained networks
    else:
        policy._l_log_std = ParamLayer(
            policy._mean_network.input_layer,
            num_units=env.spec.action_space.flat_dim,
            param=LI.Constant(np.log(init_std)),
            name='output_log_std',
            trainable=True
        )
        obs_var = policy._mean_network.input_layer.input_var
        mean_var, log_std_var = L.get_output([policy._l_mean, policy._l_log_std])
        policy._log_std_var = log_std_var
        LasagnePowered.__init__(policy, [policy._l_mean, policy._l_log_std])
        policy._f_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[mean_var, log_std_var]
        )

    safety_baseline = LinearFeatureBaseline(
        env_spec=env.spec,
        target_key='safety_returns'
    )

    safety_constraint = StraightSafetyConstraint(
        max_value=1.0,
        baseline=safety_baseline
    )

    if vv['algo'] == 'TRPO':
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=600,
            max_path_length=env.horizon,
            n_itr=10000,
            discount=0.99,
            step_size=trpo_stepsize,
            plot=False,
            # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
        )
    else:
        algo = CPO(
            env=env,
            policy=policy,
            baseline=baseline,
            safety_constraint=safety_constraint,
            batch_size=600,
            max_path_length=env.horizon,
            n_itr=600,
            discount=0.99,
            step_size=trpo_stepsize,
            gae_lambda=0.95,
            safety_gae_lambda=1,
            optimizer_args={'subsample_factor': trpo_subsample_factor},
            plot=False
        )
    algo.train()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['trpo', 'cpo'],
            default='trpo', help='Type of algorithm to use to train agent')
    parser.add_argument('--network', type=str,
            help='Path to snapshot file of pre-trained network')
    args = parser.parse_args()
    return args


def main():
    global policy
    global baseline

    # Load pre-trained network if available
    args = parse_arguments()
    if args.network is not None:
        data = joblib.load(args.network)
        policy = data['policy']
        baseline = data['baseline']
        use_pretrained = True
    else:
        use_pretrained = False

    # Run multiple experiment variants at once
    vg = VariantGenerator()

    # Non-configurable parameters (do not change)
    vg.add('trajectory', ['Straight'])
    vg.add('objective', ['TargetVelocity'])
    if args.algo == 'trpo':
        vg.add('algo', ['TRPO'])
    else:
        vg.add('algo', ['CPO'])

    # Configurable parameters
    #   Options for model_type: 'BrushTireModel', 'LinearTireModel'
    #   Options for robot_type: 'MRZR', 'RCCar'
    seeds = [102,201,54,304]
    robot_type = 'RCCar'
    use_ros = False
    vg.add('seed', seeds)
    vg.add('target_velocity', [1.0])
    vg.add('dt', [0.02])
    vg.add('model_type', ['BrushTireModel'])
    vg.add('robot_type', [robot_type])
    vg.add('mu_s', [1.37])
    vg.add('mu_k', [1.96])
    vg.add('use_ros', [use_ros])
    vg.add('pretrained', [use_pretrained])
    print('Number of Configurations: ', len(vg.variants()))

    # Run each experiment variant
    # for vv in vg.variants():
    #     run_task(vv)

    for vv in vg.variants():
        run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            n_parallel=1,
		use_gpu=True,
            snapshot_mode='last',
            seed=vv['seed']
        )
        break


if __name__ == '__main__':
    main()

