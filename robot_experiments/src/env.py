#!/usr/bin/env python
"""
title           :env.py
description     :Provides a gym-like env that interfaces with the PR2
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :01/2019
python_version  :2.7.6
==============================================================================
"""

# ROS-related imports
import rospy
import tf
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import PointCloud2, PointField

# Chainer
from chainer import serializers

# Misc
import argparse
import threading
import time
import numpy as np
import sys
import os
import os.path as osp
import math
import gym.spaces
import matplotlib.pyplot as plt

BASE_DIR = '/home/yordan/spatial_relations_experiments/'
KINECT_DIR = osp.join(BASE_DIR, 'kinect_processor')
LEARNING_DIR = osp.join(BASE_DIR, 'learning_experiments')
ROBOT_DIR = osp.join(BASE_DIR, 'robot_experiments')

sys.path.insert(0, osp.join(KINECT_DIR, 'src'))
sys.path.insert(0, osp.join(LEARNING_DIR, 'src'))
sys.path.insert(0, osp.join(ROBOT_DIR, 'src'))

# Sibling Modules
from kinect_processor import Kinect_Data_Processor
from maskrcnn_object_segmentor import Object_Segmentor
from net_200x200 import Conv_Siam_VAE
from delta_pose import PR2RobotController



class PR2Env(gym.Env):
    """Gym-like environment that interfaces the learning algorithm with the physical robot """

    def __init__(self, args=None, goal_spec=[], episode_len=10):

        self.args = args
        self.step_counter = 0
        self.episode_len = episode_len

        if self.args.render:
            fig = plt.figure()
            self.ax = fig.add_subplot(1, 1, 1)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.grid()

            # major axes
            axis_ranges = [-10, 10]
            self.ax.plot([axis_ranges[0], axis_ranges[1]], [0,0], 'k')
            self.ax.plot([0,0], [axis_ranges[0], axis_ranges[1]], 'k')
            self.ax.set_xlim(axis_ranges[0], axis_ranges[1])
            self.ax.set_ylim(axis_ranges[0], axis_ranges[1])
            
            # color map for visually encoding time
            self.cmap = plt.cm.get_cmap('cool')

        # Create action space
        self.action_space_size = 6
        low = np.ones((self.action_space_size)) * -1
        high = np.ones((self.action_space_size))
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

         # Create observation space
        self.observation_space_size = 10
        low = np.ones((self.observation_space_size)) * -30
        high = np.ones((self.observation_space_size)) * 30
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)


        # Init all necessary modules for the Kinect processing pipeline
        self.k_processor = Kinect_Data_Processor(debug=True, args=args)
        self.segmentor = Object_Segmentor(verbose=False, args=args, mode="robot")
        self.segmentor.load_model(folder_name=osp.join(KINECT_DIR, 'maskrcnn_model'), gpu_id=1)

        # TODO
        # read the groups and the group-related statistics resulting from the
        # embedding learning experiments
        groups = {0: ['front', 'behind'], 1: ['left', 'right']}
        self.embedding_model = Conv_Siam_VAE(3, 3, n_latent=8, groups=groups)
        # serializers.load_npz(osp.join(LEARNING_DIR, 'result_good_post_fix/models/final.model'), self.embedding_model)
        serializers.load_npz(osp.join(LEARNING_DIR, 'result_good_post_fix/models/final.model'), self.embedding_model)
        
        # process the incoming Kinect frames to produce a Transformed Point Cloud
        # runs in a separate thread
        self.k_processor.async_run()

        # TODO
        # calculate a goal embedding conditioned on the input goal_spec
        # TEMPORARILY sample a random goal vector
        self.goal_em = np.random.uniform(low=-5, high=5, size=2)
        self.spec_vector = self.generate_spec_vector()

        # init the robot controller
        self.pr2 = PR2RobotController('right_arm')


    def get_embedding(self):
        # get latest Transformed Poing Cloud and segment it
        # k_processor runs in a separate thread
        (xyz, bgr) = self.k_processor.output[-1]
        self.segmentor.load_processed_frame([xyz, bgr])
        self.segmentor.process_data()

        # embed the segmented Point Cloud
        # !! swap the axes for the object PCs before feeding into the model
        b0 = self.segmentor.output[0]['green_cube']
        b1 = self.segmentor.output[0]['red_cube']
        




        # every_n = 15
        # fig = plt.figure(figsize=(8,8))
        # ax = fig.add_subplot(1, 1, 1, projection='3d')

        # xs = b0[..., 0][::every_n]
        # ys = b0[..., 1][::every_n]
        # zs = b0[..., 2][::every_n]
        # ax.scatter(xs, ys, zs, c='r', alpha=0.5)

        # xs = b1[..., 0][::every_n]
        # ys = b1[..., 1][::every_n]
        # zs = b1[..., 2][::every_n]
        # ax.scatter(xs, ys, zs, c='c', alpha=0.1)

        # ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        # ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        # ax.zaxis.set_major_locator(plt.MaxNLocator(2))

        # ax.set_xlabel('Z0/X', fontweight="bold")
        # ax.set_ylabel('Z1/Y', fontweight="bold")
        # ax.set_zlabel('Z2/Z', fontweight="bold")




        b0 = b0[np.newaxis, :]
        b0 = np.swapaxes(b0, 1, 3).astype(np.float32)
        b1 = b1[np.newaxis, :]
        b1 = np.swapaxes(b1, 1, 3).astype(np.float32)
        embedding = self.embedding_model.get_latent(b0, b1).data[0]
        print(embedding)
        # embedding = np.array([x.data for x in embedding]).astype(np.float32)

        # plt.show()

        return embedding


    def get_state(self, target_frame="base_link", ee_frame="/r_gripper_r_finger_tip_link"):
        # return obs vector - [x, y, z, xr, yr, zr, obs_em, goal_em]
        self.k_processor.tf_listener.waitForTransform(target_frame, ee_frame, rospy.Time(), rospy.Duration(4.0))
        position, quaternion = self.k_processor.tf_listener.lookupTransform(target_frame, ee_frame, rospy.Duration(0))
        orientation = euler_from_quaternion (quaternion)

        obs_em = self.get_embedding()
        self.obs_em = obs_em

        return np.concatenate((position, orientation, obs_em.flatten(), self.goal_em.flatten())).astype(np.float32)


    # return the reward, given a goal embedding and an observation embedding
    # we get highest reward when the two vectors are the same
    def get_reward(self, goal_em=None, obs_em=None, scaling_factor=1):
        cosine_sim = np.dot(goal_em, obs_em) / (np.linalg.norm(goal_em) * np.linalg.norm(obs_em))
        magnitude_diff = abs(np.linalg.norm(goal_em) - np.linalg.norm(obs_em))

        return scaling_factor * (cosine_sim - magnitude_diff)


    # given an action, execute it on the robot and return (next state, reward)
    def step(self, a):

        # do action a (sent to robot and wait)
        (delta_t, delta_rpy) = a
        self.pr2.move_delta_t_rpy(delta_t, delta_rpy)
        # self.pr2.move_delta_t_rpy(np.zeros(3), np.zeros(3))

        s = self.get_state()
        r = self.get_reward(goal_em=s[-2], obs_em=s[-1])

        done = False

        if self.step_counter >= self.episode_len:
            done = self.check_spec_satisfaction()
        else:
            self.step_counter += 1

        return (s.flatten().astype(np.float32), r, done, None)


    def reset(self):
        self.pr2.reset_pose()
        self.step_counter = 0
        time.sleep(3)

        return self.get_state().flatten().astype(np.float32)


    def render(self):
        c_index = self.step_counter / float(self.episode_len)
        print(c_index)
        self.ax.scatter(self.obs_em[0], self.obs_em[1], color=self.cmap(c_index), marker='o', s=50, alpha=0.75)
        plt.draw()
        plt.pause(0.5)


    def generate_spec_vector(self):
        result = np.zeros(3)

        # spec -> {label: [index, sign]}
        spec = {}

        # X
        spec['front'] = [0, -1]
        spec['behind'] = [0, 1]

        # Y
        spec['left'] = [1, -1]
        spec['right'] = [1, 1]

        # Z
        spec['below'] = [2, -1]
        spec['above'] = [2, 1]

        for label in self.goal_spec:
            axis_index = spec[label][0]
            result[axis_index] = spec[label][1]

        return result


    def check_spec_satisfaction(self):

        done = True

        (xyz, bgr) = self.k_processor.output[-1]
        self.segmentor.load_processed_frame([xyz, bgr])
        self.segmentor.process_data()

        # embed the segmented Point Cloud
        # !! swap the axes for the object PCs before feeding into the model
        b0 = self.segmentor.output[0]['green_cube']
        b1 = self.segmentor.output[0]['red_cube']

        b0_m = np.mean(b0, axis=(0,1))
        b1_m = np.mean(b1, axis=(0,1))

        diff = np.sign(b0_m - b1_m)

        for i, entry in enumerate(self.spec_vector):
            if entry != 0:
                done = done and diff[i] == entry

        return done


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Save Kinect data to NPZ file')
    parser.add_argument('--cutoff', default=1000, type=int, 
                        help='Number of frames to be captured')
    parser.add_argument('--scene', '-sc', default='100', 
                        help='Index for a scene/setup')
    parser.add_argument('--mode', '-m', default='gather',
                        help='Whether we are gathering data or learning')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbose logging')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the env')
    args = parser.parse_args()

    env = PR2Env(args=args)
    s = env.reset()

    for _ in range(10):
        env.render()

        # delta_t = np.random.uniform(low=-0.02, high=0.02, size=3)
        # delta_rpy = np.random.uniform(low=-math.pi/12., high=math.pi/12., size=3)
        delta_t = np.zeros(3)
        delta_t[0] = 0.05
        delta_rpy = np.zeros(3)
        print("Delta t:\t {0},\tDelta RPY:\t {1}".format(delta_t, delta_rpy))
        
        env.step([delta_t, delta_rpy])

    plt.show()

    # final cleanup
    env.k_processor.async_close()

    print("END OF FILE")

