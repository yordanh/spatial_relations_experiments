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

# BASE_DIR = '/home/yordan/spatial_relations_experiments/'
BASE_DIR = '/home/yordan/pr2_ws/src/spatial_relations_experiments/'
KINECT_DIR = osp.join(BASE_DIR, 'kinect_processor')
LEARNING_DIR = osp.join(BASE_DIR, 'learning_experiments')
ROBOT_DIR = osp.join(BASE_DIR, 'robot_experiments')
RESULT_DIR = osp.join(LEARNING_DIR, 'result/full')

sys.path.insert(0, osp.join(KINECT_DIR, 'src'))
sys.path.insert(0, osp.join(LEARNING_DIR, 'src'))
sys.path.insert(0, osp.join(ROBOT_DIR, 'src'))

# Sibling Modules
from kinect_processor import Kinect_Data_Processor
from maskrcnn_object_segmentor import Object_Segmentor
from net_100x100 import Conv_Siam_VAE
from delta_pose import PR2RobotController



class PR2Env(gym.Env):
    """Gym-like environment that interfaces the learning algorithm with the physical robot """

    def __init__(self, args=None, goal_spec=None, objects=['green_cube', 'blue_die'], episode_len=None, group_n=None, steps=None):

        self.args = args
        self.step_counter = 0
        self.episode_len = episode_len
        self.objects = objects
        self.b0 = None
        self.b1 = None
        self.last_s = None
        self.training_stats = [[]]
        self.diffs = []
        self.steps = steps
        self.update_both = True

        if self.args.render:
            fig = plt.figure()
            self.ax = fig.add_subplot(1, 1, 1)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.grid()

            # major axes
            axis_ranges = [-5, 5]
            self.ax.plot([axis_ranges[0], axis_ranges[1]], [0,0], 'k')
            self.ax.plot([0,0], [axis_ranges[0], axis_ranges[1]], 'k')
            # self.ax.set_xlim(axis_ranges[0], axis_ranges[1])
            # self.ax.set_ylim(axis_ranges[0], axis_ranges[1])
            
            # color map for visually encoding time
            self.cmap = plt.cm.get_cmap('cool')

        # Init all necessary modules for the Kinect processing pipeline
        self.k_processor = Kinect_Data_Processor(debug=True, args=args, mode="learning")
        self.segmentor = Object_Segmentor(verbose=False, args=args, mode="robot")
        self.segmentor.load_model(folder_name=osp.join(KINECT_DIR, 'maskrcnn_model'), gpu_id=args.gpu)


        self.k_processor.tf_listener.waitForTransform("base_link", "/r_gripper_r_finger_tip_link", rospy.Time(), rospy.Duration(4.0))
        position_r, quaternion_r = self.k_processor.tf_listener.lookupTransform("base_link", "/r_gripper_r_finger_tip_link", rospy.Duration(0))
        self.position_r = position_r


        # TODO
        # read the groups and the group-related statistics resulting from the
        # embedding learning experiments
        groups = {0: ['front', 'behind'], 1: ['left', 'right']}
        self.embedding_model = Conv_Siam_VAE(3, 3, n_latent=8, groups=groups)
        serializers.load_npz(osp.join(RESULT_DIR, 'models/final.model'), self.embedding_model)
        self.embedding_model.to_cpu()
        
        # process the incoming Kinect frames to produce a Transformed Point Cloud
        # runs in a separate thread
        self.k_processor.async_run()

        # Create action space
        self.action_space_size = 2
        low = np.ones((self.action_space_size)) * -1
        high = np.ones((self.action_space_size))
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

         # Create observation space
        if self.args.state_type == "eef":
            self.observation_space_size = 4
        elif self.args.state_type == "embed":
            self.observation_space_size = 2

        low = np.ones((self.observation_space_size)) * -1
        high = np.ones((self.observation_space_size)) * 1
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        # TODO
        # calculate a goal embedding conditioned on the input goal_spec
        # TEMPORARILY sample a random goal vector
        self.goal_spec = goal_spec
        self.spec_vector = self.generate_spec_vector()

        # SPEC = ['right']
        self.goal_em = np.zeros(2)
        # self.goal_em[0] = np.random.uniform(low=-2, high=2, size=1)
        # self.goal_em[1] = np.random.uniform(low=-2, high=-0.5, size=1)
        self.goal_em = np.array([1, -1], dtype=np.float32)

        print(self.args.active_arm)
        # init the robot controller
        self.pr2 = PR2RobotController(str(self.args.active_arm))
        # self.pr2 = PR2RobotController('right_arm')

        self.ref_object_posit = np.array([0.65, -0.05])
        # self.ref_object_posit = np.array([0.7, -0.15])

        # behind, right
        self.euclid_goal_posit = self.ref_object_posit + np.array([0.15, -0.15])
        


        self.update_pc(update_both=True)
        time.sleep(2)


    def sample_goal_em(self):
        
        # for index, entry in enumerate(self.spec_vector):
        #     if entry == 0:
        self.goal_em[0] = np.random.uniform(low=0.5, high=1.5, size=1)
        self.goal_em[1] = np.random.uniform(low=-1.5, high=-0.5, size=1)

        print("goal_em", self.goal_em)


    def get_embedding(self):

        b0 = self.b0[np.newaxis, :]
        b0 = np.swapaxes(b0, 1, 3).astype(np.float32)
        b1 = self.b1[np.newaxis, :]
        b1 = np.swapaxes(b1, 1, 3).astype(np.float32)
        embedding, _ = self.embedding_model.get_latent_pred(b0, b1)
        embedding = embedding.data[0]

        return embedding


    def get_state(self, target_frame="base_link"):
        # return obs vector - [x, y, z, xr, yr, zr, <obs_em>, <goal_em>] * 2 (for both arms)
        # the embeddings are optional depending on self.args.state_type
        self.k_processor.tf_listener.waitForTransform(target_frame, "/r_gripper_r_finger_tip_link", rospy.Time(), rospy.Duration(4.0))
        position_r, quaternion_r = self.k_processor.tf_listener.lookupTransform(target_frame, "/r_gripper_r_finger_tip_link", rospy.Duration(0))
        orientation_r = euler_from_quaternion (quaternion_r)

        # self.k_processor.tf_listener.waitForTransform(target_frame, "/l_gripper_l_finger_tip_link", rospy.Time(), rospy.Duration(4.0))
        # position_l, quaternion_l = self.k_processor.tf_listener.lookupTransform(target_frame, "/l_gripper_l_finger_tip_link", rospy.Duration(0))
        # orientation_l = euler_from_quaternion (quaternion_l)

        if self.args.state_type == "eef":
            # result = np.concatenate((position_l, orientation_l, position_r, orientation_r))
            result = np.concatenate((self.position_r[:2], self.ref_object_posit))
        
        elif self.args.state_type == "embed":
            self.obs_em = self.get_embedding()

            # print("Obs Em -> {0}".format(self.obs_em))
            # print("Goal Em -> {0}".format(self.goal_em))

            # result = np.concatenate((position_l, orientation_l, position_r, orientation_r, self.obs_em.flatten(), self.goal_em.flatten()))
            # result = np.concatenate((position_r[:2], orientation_r, self.obs_em.flatten(), self.goal_em.flatten()))
            # result = np.concatenate((self.obs_em.flatten(), self.goal_em.flatten()))
            result = self.obs_em.flatten()

        return result.flatten()


    # return the reward, given a goal embedding and an observation embedding
    # we get highest reward when the two vectors are the same
    def get_reward(self, scaling_factor=1):
        
        if self.args.reward_type == "euclid":
            a = self.position_r[:2] - self.ref_object_posit
            b = self.euclid_goal_posit - self.ref_object_posit
            result = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            return result

        elif self.args.reward_type == "euclid_dist":
            result =  1 - (np.linalg.norm(self.position_r[:2] - self.euclid_goal_posit) / 0.5)
            return result

        elif self.args.reward_type == "discrete":
            result = int(self.check_spec_satisfaction()[0])
            if result == 0:
                result = -1

            result = scaling_factor * result

        elif self.args.reward_type == "embed":
            
            self.obs_em = self.get_embedding()

            # if self.args.state_type != "embed":
                # print("Obs Em -> {0}".format(self.obs_em))
                # print("Goal Em -> {0}".format(self.goal_em))

            cosine_sim = np.dot(self.goal_em, self.obs_em) / (np.linalg.norm(self.goal_em) * np.linalg.norm(self.obs_em))
            magnitude_diff = abs(np.linalg.norm(self.goal_em) - np.linalg.norm(self.obs_em))
            # result = scaling_factor * (cosine_sim - magnitude_diff)
            result = scaling_factor * (cosine_sim)

        return result


    # given an action, execute it on the robot and return (next state, reward)
    def step(self, a):
        # do action a (sent to robot and wait)
        delta_t = np.zeros(3)
        delta_t[0] = a[0] / 25.
        delta_t[1] = a[1] / 25.

        delta_rpy = np.zeros(3)

        self.b0 += delta_t
        self.b0[self.b0 == delta_t] = 0
        self.position_r += delta_t

        # print("Delta_t", delta_t)

        self.step_counter += 1
               
        done = False
        # spec_satisfaction = self.check_spec_satisfaction()
        spec_satisfaction = False

        s = self.get_state()
        r = self.get_reward()

        # self.show_pcs()
        
        self.last_s = s

        self.training_stats[-1].append(r)


        if self.step_counter >= self.episode_len:

            spec_satisfaction, diff = self.check_spec_satisfaction()
            self.diffs.append(diff)

            print("STEP {3}, Done: {0}\tSpec Sat: {1}\tR: {2}".format(done, spec_satisfaction, r, self.step_counter))

            output = {"stats" : self.training_stats, "episodes_n" : self.steps / self.episode_len, "diffs" : self.diffs}
            np.savez(os.path.join(self.args.outdir, "training_stats.npz"), **output)
            self.training_stats.append([])
            done = True


        return (s, r, done, {'spec_satisfaction': spec_satisfaction, 'needs_reset': done})


    def reset(self):
        print('\n##### RESET #####')

        x = np.random.uniform(low=0.3, high=0.9)
        y = np.random.uniform(low=-0.3, high=0.3)

        # wrist offset
        y -= 0.1

        # print("X, Y", x, y)

        # 5cm buffer
        while((x > 0.45 and y < 0.1)):
            x = np.random.uniform(low=0.3, high=0.9)
            y = np.random.uniform(low=-0.3, high=0.3)
            y -= 0.1
            # print("X, Y Resampled", x, y)

        new_position = np.array([x,y,0])
        self.position_r = new_position

        b0 = self.b0[self.b0 != [0,0,0]]
        b0 = b0.reshape(len(b0) / 3, 3)
        b0_m = np.mean(b0, axis=0)

        # print("b0_m", np.round(b0_m, decimals=3))

        bounds = {'x':[0.1, 1.3], 'y':[-0.8, 0.8], 'z':[0.4, 1.4]}
        x_norm = (x - bounds['x'][0]) / (bounds['x'][1] - bounds['x'][0])
        y_norm = (y - bounds['y'][0]) / (bounds['y'][1] - bounds['y'][0])

        offset = np.array([x_norm,y_norm,0]) - b0_m
        offset[2] = 0

        # print("offset", np.round(offset, decimals=3))

        self.b0 += offset
        self.b0[self.b0 == offset] = 0


        b0 = self.b0[self.b0 != [0,0,0]]
        b0 = b0.reshape(len(b0) / 3, 3)
        b0_m = np.mean(b0, axis=0)

        print("b0_m post reset", np.round(b0_m, decimals=3))


        self.last_s = np.zeros(self.observation_space_size)
        self.step_counter = 0

        if self.update_both == True:
            self.update_both = False


        # self.sample_goal_em()

        return self.get_state()


    def render(self, mode):
        c_index = self.step_counter / float(self.episode_len)
        self.ax.scatter(self.obs_em[0], self.obs_em[1], color=self.cmap(c_index), marker='.', s=50, alpha=0.75)
        plt.draw()
        plt.pause(0.1)


    def generate_spec_vector(self):
        result = np.zeros(2)

        # spec -> {label: [index, sign]}
        spec = {}

        # X
        spec['front'] = [0, -1]
        spec['behind'] = [0, 1]

        # Y
        spec['left'] = [1, 1]
        spec['right'] = [1, -1]

        # Z
        # spec['below'] = [2, -1]
        # spec['above'] = [2, 1]

        if self.goal_spec is not None:
            for label in self.goal_spec:
                axis_index = spec[label][0]
                result[axis_index] = spec[label][1]

        return result


    def check_spec_satisfaction(self):

        spec_satisfaction = True

        
        b0 = self.b0[self.b0 != [0,0,0]]
        b0 = b0.reshape(len(b0) / 3, 3)
        b1 = self.b1[self.b1 != [0,0,0]]
        b1 = b1.reshape(len(b1) / 3, 3)

        b0_m = np.mean(b0, axis=0)
        b1_m = np.mean(b1, axis=0)

        diff = np.sign(b0_m - b1_m)
        print("b0_m", b0_m)
        print("b1_m", b1_m)
        print("Diff", b0_m - b1_m)
        # print("\n")

        for i, entry in enumerate(self.spec_vector):
            if entry != 0:
                spec_satisfaction = spec_satisfaction and diff[i] == entry

        return spec_satisfaction, (b0_m - b1_m)


    def update_pc(self, update_both=False):
        time.sleep(0.5)
        needs_reset = False
        pc_update_success = False

        while not pc_update_success:
            # get latest Transformed Poing Cloud and segment it
            # k_processor runs in a separate thread
            (xyz, bgr) = self.k_processor.output[-1]
            self.segmentor.load_processed_frame([xyz, bgr])

            if update_both:
                objects = self.objects
            else:
                objects = [self.objects[0]]

            try:
                self.segmentor.process_data(objects)
            except:
                print("\nSEGMENTOR CAN'T PROCESS DATA")
                needs_reset = True
                return needs_reset

            try:
                # embed the segmented Point Cloud
                # self.objects[<b0_object_name>, <b1_object_name>]
                # !! swap the axes for the object PCs before feeding into the model
                self.b0 = self.segmentor.output[-1][self.objects[0]]
                
                if update_both:
                    self.b1 = self.segmentor.output[-1][self.objects[1]]

                pc_update_success = True
            except:
                print("\nCAN'T SEGMENT WELL, ONLY SEGMENTS {0} \n".format(self.segmentor.output[-1]))
                needs_reset = True
                return needs_reset

        return needs_reset


    def show_pcs(self):

        b0 = self.b0[self.b0 != [0,0,0]]
        b0 = b0.reshape(len(b0) / 3, 3)
        b0_m = np.mean(b0, axis=0)

        b1 = self.b1[self.b1 != [0,0,0]]
        b1 = b1.reshape(len(b1) / 3, 3)
        b1_m = np.mean(b1, axis=0)


        # print("b0_m", b0_m)
        # print("b1_m", b1_m)


        every_n = 15
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        xs = b0[..., 0][::every_n]
        ys = b0[..., 1][::every_n]
        zs = b0[..., 2][::every_n]
        ax.scatter(xs, ys, zs, c='r', alpha=0.5)

        xs = b1[..., 0][::every_n]
        ys = b1[..., 1][::every_n]
        zs = b1[..., 2][::every_n]
        ax.scatter(xs, ys, zs, c='c', alpha=0.1)

        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        ax.zaxis.set_major_locator(plt.MaxNLocator(2))

        ax.set_xlabel('Z0/X', fontweight="bold")
        ax.set_ylabel('Z1/Y', fontweight="bold")
        ax.set_zlabel('Z2/Z', fontweight="bold")
        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Save Kinect data to NPZ file')
    parser.add_argument('--cutoff', default=1000, type=int, 
                        help='Number of frames to be captured')
    parser.add_argument('--scene', '-sc', default='100', 
                        help='Index for a scene/setup')
    parser.add_argument('--mode', '-m', default='gather',
                        help='Whether we are gathering data or learning')
    parser.add_argument('--state_type', default='eef',
                        help='The type of state space used by the agent - [eef, eff_embed]')
    parser.add_argument('--reward_type', default='discrete',
                        help='The type of reward used by the agent - [discrete, embed]')
    parser.add_argument('--active_arm', default="right_arm", type=str,
                        help='The arm which will be actucated - [right_arm, left_arm]')
    parser.add_argument('--task_name', default='right',
                        help='The name for the spatial prep for which a policy is learned - [left, right, front, behind]')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbose logging')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the env')
    args = parser.parse_args()

    env = PR2Env(args=args, goal_spec=['behind'])
    s = env.reset()

    for i in range(10):
        for j in range(10):
            print(i,j)
            # env.update_pc()
            # env.obs_em = env.get_embedding()

            delta = np.random.uniform(low=-1, high=1, size=6)
            # delta_rpy = np.random.uniform(low=-math.pi/12., high=math.pi/12., size=3)
            # delta = np.zeros(6)
            # delta[0] = 0.5
            # delta_rpy = np.zeros(3)
            # print("Delta :\t {0}".format(delta))
            
            result = env.step(delta)
            print(result[3])
            if result[3]['needs_reset']:
                break

            env.render()

        env.reset()

    plt.show()

    # final cleanup
    env.k_processor.async_close()

    print("END OF FILE")

