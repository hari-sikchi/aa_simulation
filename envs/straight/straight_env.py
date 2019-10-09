#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Environment for training a local planner to move in a straight line.
"""

import numpy as np
import time

from rllab.spaces import Box

from aa_simulation.envs.base_env import VehicleEnv
from aa_simulation.misc.utils import normalize_angle

_old_action = None
_prev_x = None
class StraightEnv(VehicleEnv):
    """
    Simulation environment for an RC car following a straight
    line trajectory. The reward function encourages the agent to
    move right on the line y=0 for all time.
    """

    def __init__(
            self,
            target_velocity=1.0,
            dt=0.035,
            model_type='BrushTireModel',
            robot_type='RCCar',
            mu_s=1.37,
            mu_k=1.96
           
    ):
        """
        Initialize super class parameters, obstacles and radius.
        """
        # self.pending_actions = []
        # # action drop probablity from the pending action_queue
        # self.action_drop_prob = 0.2
        # self.old_time = time.time()
        # self.old_action = None

        super(StraightEnv, self).__init__(
            target_velocity=target_velocity,
            dt=dt,
            model_type=model_type,
            robot_type=robot_type,
            mu_s=mu_s,
            mu_k=mu_k
        )
        self.robot_type = robot_type

        # Reward function parameters
        self._lambda1 = 0.25


    @property
    def observation_space(self):
        """
        Define the shape of input vector to the neural network.
        """
        return Box(low=-np.inf, high=np.inf, shape=(5,))


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.
        """
        # Randomly initialize state for better learning
        global _prev_x,_old_action
        _prev_x=None
        _old_action=None
        if self.robot_type == 'RCCar':
            y = np.random.uniform(-0.25, 0.25)
            # y=0
            # yaw = 0
            yaw = np.random.uniform(-np.pi/3, np.pi/3)
            #self._dt =np.random.uniform(0.01, 0.05)  
            x_dot = np.random.uniform(0, 1.3)
            y_dot = np.random.uniform(-0.6, 0.6)
            yaw_dot = np.random.uniform(-2.0, 2.0)
        elif self.robot_type == 'MRZR':
            y = np.random.uniform(-0.25, 0.25)
            yaw = np.random.uniform(-np.pi/3, np.pi/3)
            x_dot = np.random.uniform(0, 2.0)
            y_dot = np.random.uniform(-0.6, 0.6)
            yaw_dot = np.random.uniform(-0.3, 0.3)
        else:
            raise ValueError('Unrecognized robot type')

        state = np.zeros(6)
        state[1] = y
        state[2] = yaw
        state[3] = x_dot
        state[4] = y_dot
        state[5] = yaw_dot
        return state


    def get_reward(self, state, action,prev_state=None):
        """
        Reward function definition.
        """
        global _old_action,_prev_x
        # print(action)
        x, y, _, x_dot, y_dot, _ = state
        velocity = np.sqrt(x_dot**2 + y_dot**2)
        distance = y
        if(_old_action is None):
            _old_action=action
        if(_prev_x is None):
            _prev_x=x        
        # penalty for action divergence
        reward = -np.absolute(distance)
        alpha = 5
        if prev_state is not None:
            div_action =  - np.sum(np.square(prev_state[2] - state[2]))
            reward+=alpha*div_action

        #div_action =  - np.sum(np.square(action[1] - _old_action[1]))  
        forward_motion = np.maximum(x-_prev_x,0)
        beta = 10
        reward -= self._lambda1 * (velocity - self.target_velocity)**2
        # reward +=beta * forward_motion
        # print(beta * forward_motion,np.absolute(distance),self._lambda1 * (velocity - self.target_velocity)**2,alpha*div_action)
        #   print(reward)
        _old_action=action
        _prev_x=x


        info = {}
        info['dist'] = distance
        info['vel'] = velocity
        info['reward'] = reward
        return reward, info


    @staticmethod
    def project_line(state, x0, y0, angle):
        """
        Note that this policy is trained to follow a straight line
        to the right (y = 0). To follow an arbitrary line, use this
        function to transform the current absolute state to a form
        that makes the policy believe the car is moving to the right.

        :param state: Current absolute state of robot
        :param x0: x-value of start of line to follow
        :param y0: y-value of start of line to follow
        :param angle: Angle of line to follow
        """
        x, y, yaw, x_dot, y_dot, yaw_dot = state
        angle = normalize_angle(angle)

        current_angle = np.arctan2((y - y0), (x - x0))
        projected_angle = normalize_angle(current_angle - angle)
        dist = np.sqrt((x - x0)**2 + (y - y0)**2)

        new_x = dist * np.cos(projected_angle)
        new_y = dist * np.sin(projected_angle)
        new_yaw = normalize_angle(yaw - angle)

        return np.array([new_x, new_y, new_yaw, x_dot, y_dot, yaw_dot])


    def state_to_observation(self, state):
        """
        Prepare state to be read as input to neural network.
        """
        _, y, yaw, x_dot, y_dot, yaw_dot = state
        yaw = normalize_angle(yaw)
        
        #return np.array([y, yaw, x_dot,self._dt])
        # return np.array([y, yaw, x_dot, y_dot, yaw_dot])
        return np.array([y, yaw, x_dot, y_dot, yaw_dot])

