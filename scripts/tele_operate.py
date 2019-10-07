#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Evaluate a policy and publish metrics.
"""

import argparse
import cProfile
import pstats
import sys
import numpy as np
from rllab.misc import tensor_utils
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/harshit/work/rllab')
from rllab.sampler.utils import rollout
from aa_simulation.envs.base_env import VehicleEnv
from pynput.keyboard import Key, Listener
action_teleop = np.array([1.00,0.0001])

def on_press(key):
    global action_teleop
    print('{0} pressed'.format(
        key))
    if(key=='w'):
        action_teleop[0]+=1
    if(key=='s'):
        action_teleop[0]-=1
    if(key=='d'):
        action_teleop[1]+=0.1
    if(key=='a'):
        action_teleop[1]-=0.1    
    

def on_release(key):
    print('{0} release'.format(
        key))
    if key == Key.esc:
        # Stop listener
        return False








# Toggle option for displaying plots
show_plots = True

# Statistics over multiple runs
means_speed = []
means_steer = []
means_slip = []
means_dist = []
means_vel = []


def _find_getch():
    try:
        import termios
    except ImportError:
        # Non-POSIX. Return msvcrt's (Windows') getch.
        import msvcrt
        return msvcrt.getch

    # POSIX system. Create and return a getch that manipulates the tty.
    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    return _getch



def rollout_teleop(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False):
    global action_teleop
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        print(action_teleop)
        next_o, r, d, env_info = env.step(action_teleop)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )


def profile_code(profiler):
    """
    Use cProfile to profile code, listing functions with most
    cumulative time spent.
    """
    print('\n')
    ps = pstats.Stats(profiler).strip_dirs().sort_stats('cumulative')
    ps.print_stats(10)


def plot_curve(data, name, units):
    """
    Plot data over time.
    """
    mean = data.mean()
    std = data.std()
    maximum = data.max()
    minimum = data.min()
    stats = 'Mean = %.5f\nStd = %.5f\nMax = %.5f\nMin = %.5f' % \
            (mean, std, maximum, minimum)
    title = '%s over Time in Final Policy' % name

    if not show_plots:
        return

    plt.figure()
    t = np.arange(data.size)
    plt.plot(t, data)
    plt.title(title)
    plt.xlabel('Time steps')
    plt.ylabel('%s (%s)' % (name, units))
    plt.axhline(mean, color='k', linestyle='dashed', linewidth=1)
    plt.axhline(mean+std, color='r', linestyle='dashed', linewidth=1)
    plt.axhline(mean-std, color='r', linestyle='dashed', linewidth=1)
    plt.text(0.87, 0.9, stats, ha='center', va='center',
            transform=plt.gca().transAxes)


def plot_distribution(data, name, units):
    """
    Plot histogram showing distribution of data.
    """
    mean = data.mean()
    std = data.std()
    maximum = data.max()
    minimum = data.min()
    stats = 'Mean = %.5f\nStd = %.5f\nMax = %.5f\nMin = %.5f' % \
            (mean, std, maximum, minimum)
    title = 'Distribution of %s in Final Policy' % name

    if not show_plots:
        return

    plt.figure()
    plt.hist(data)
    plt.title(title)
    plt.xlabel('Error (%s)' % units)
    plt.ylabel('Number of Time Steps')
    plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(mean+std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(mean-std, color='r', linestyle='dashed', linewidth=1)
    plt.text(0.87, 0.9, stats, ha='center', va='center',
            transform=plt.gca().transAxes)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='Path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=250,
                        help='Max length of rollout')
    parser.add_argument('--seed', type=int, default=9, help='Random seed')
    parser.add_argument('--speedup', type=float, default=100000,
                        help='Speedup')
    parser.add_argument('--skip', type=int, default=0,
                        help='Number of iterations to skip at start')
    parser.add_argument('--num_paths', type=int, default=1,
                        help='Number of rollouts to collect and evaluate')
    parser.add_argument('--render', dest='render',
            action='store_true', help='Rendering')
    parser.add_argument('--no-render', dest='render',
            action='store_false', help='Rendering')
    parser.set_defaults(render=False)
    parser.add_argument('--plots', dest='show_plots',
            action='store_true', help='Show plots')
    parser.add_argument('--no-plots', dest='show_plots',
            action='store_false', help='Show plots')
    parser.set_defaults(show_plots=True)
    parser.add_argument('--profile', dest='profile_code',
            action='store_true', help='Profile code that samples a rollout')
    parser.add_argument('--no-profile', dest='profile_code',
            action='store_false', help='Profile code that samples a rollout')
    parser.set_defaults(profile_code=False)
    args = parser.parse_args()
    return args


def main():
    global show_plots
    global means_speed
    global means_steer
    global means_slip
    global means_dist
    global means_vel

    args = parse_arguments()
    profiler = cProfile.Profile()
    data = joblib.load(args.file)
    skip = args.skip
    policy = data['policy']
    env = data['env']
    env._dt = 0.035                # Set dt to empirically measured dt
    np.random.seed(args.seed)
    show_plots = args.show_plots
    if show_plots:
        plt.ion()

    print("num_paths : {}".format(args.num_paths))
    print("skip : {}".format(skip))

    # Collect events until released
    # Listener(on_press=on_press)
    # with Listener(
    #         on_press=on_press) as listener:
    #     pass
        # listener.join()

    for run in range(args.num_paths):

        profiler.enable()
        path = rollout_teleop(env, policy, max_path_length=args.max_path_length,
                            animated=args.render, speedup=args.speedup,
                            always_return_paths=True)
        profiler.disable()
        if args.profile_code:
            profile_code(profiler)

        # Analyze rollout
        actions = path['actions']
        plot_curve(actions[:, 0][skip:], 'Commanded Speed', 'm/s')
        plot_curve(actions[:, 1][skip:], 'Commanded Steering Angle', 'rad')
        plot_curve(path['env_infos']['kappa'][skip:], 'Wheel Slip', 'kappa')
        plot_curve(path['env_infos']['dist'][skip:], 'Distance', 'm')
        plot_curve(path['env_infos']['vel'][skip:], 'Velocity', 'm/s')
        plot_distribution(path['env_infos']['dist'][skip:], 'Distance', 'm')
        plot_distribution(path['env_infos']['vel'][skip:], 'Velocity', 'm/s')

        means_speed.append(actions[:, 0][skip:].mean())
        means_steer.append(actions[:, 1][skip:].mean())
        means_slip.append(path['env_infos']['kappa'][skip:].mean())
        means_dist.append(path['env_infos']['dist'][skip:].mean())
        means_vel.append(path['env_infos']['vel'][skip:].mean())
        print("Mean abs vel: {}".format(np.mean(np.abs(path['env_infos']['vel'][skip:]))))
        print("Mean abs distance: {}".format(np.mean(np.abs(path['env_infos']['dist'][skip:]))))

    # Print statistics over multiple runs
    if not args.profile_code:
        print()
    means_speed = np.array(means_speed)
    means_steer = np.array(means_steer)
    means_slip = np.array(means_slip)
    means_dist = np.array(means_dist)
    means_vel = np.array(means_vel)
    
    print('Averaged statistics over %d rollout(s):' % args.num_paths)
    print('\tMean Commanded Speed:\t%.5f +/- %.5f'
            % (means_speed.mean(), means_speed.std()))
    print('\tMean Commanded Steer:\t%.5f +/- %.5f'
            % (means_steer.mean(), means_steer.std()))
    print('\tMean Slip:\t\t%.5f +/- %.5f'
            % (means_slip.mean(), means_slip.std()))
    print('\tMean Distance Error:\t%.5f +/- %.5f'
            % (means_dist.mean(), means_dist.std()))
    print('\tMean Velocity:\t\t%.5f +/- %.5f'
            % (means_vel.mean(), means_vel.std()))
    print()

    plt.show()

    # Block until key is pressed
    sys.stdout.write("Press <enter> to continue: ")
    input()


if __name__ == "__main__":
    main()

