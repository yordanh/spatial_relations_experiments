#!/usr/bin/env python
"""
title           :spatial_augmenter.py
description     :Augments spatially - xyz variations - the pointcloud it is given
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :12/2018
python_version  :2.7.6
==============================================================================
"""
import numpy as np
import cv2
import random, math, os
import os.path as osp
import copy
import sys
import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import shutil

# seed = 0
# np.random.seed(seed)


class SpatialAugmenter():
    def __init__(self, random_seed=0, verbose=0):
        self.verbose=verbose

        # 2 cm
        self.noise = 0.01

    def augment(self, branch_0, branch_1):

        result = [[],[]]
        
        for pair in zip(branch_0, branch_1):
            pair = np.array(pair)
            points = pair.copy()

            global_offsets = np.zeros((3))

            ranges = []
            ranges.append(0 - np.min(points, axis=(0,1,2)))
            ranges.append(1 - np.max(points, axis=(0,1,2)))
            ranges[0][2] = 0

            for i in range(len(global_offsets)):
                global_offsets[i] = np.random.uniform(low=ranges[0][i], high=ranges[1][i])
                
            # print(global_offsets)

            # overall offset
            points += global_offsets
            points[points == global_offsets] = 0

            # individual object offsets
            for i in range(len(points)):
                local_offsets = np.zeros((3))
                for j in range(len(local_offsets)):
                    local_offsets[j] = np.random.uniform(low=-self.noise, high=self.noise)
                points[i] += local_offsets
                points[i][points[i] == local_offsets] = 0

#                 print(local_offsets)
                
            for i in range(len(points)):
                result[i].append(points[i])

        return np.array(result)

    def plot(self, data_orig, data_augmented):
    
        data_orig = np.array(data_orig)
        data_augmented = np.array(data_augmented)
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        points = data_orig.reshape(np.product(data_orig.shape[:-1]), 3)
        filtered_points = np.array(list(filter(lambda row : filter(lambda point : (point != [0,0,0]).all(), row), points)))

        xs = filtered_points[...,0][::3]
        ys = filtered_points[...,1][::3]
        zs = filtered_points[...,2][::3]

        ax.scatter(xs, ys, zs, c='c')

        points = data_augmented.reshape(np.product(data_augmented.shape[:-1]), 3)
        filtered_points = np.array(list(filter(lambda row : filter(lambda point : (point != [0,0,0]).all(), row), points)))

        xs = filtered_points[...,0][::3]
        ys = filtered_points[...,1][::3]
        zs = filtered_points[...,2][::3]

        ax.scatter(xs, ys, zs, c='r')

        ax.set_xlabel('X', fontsize='20', fontweight="bold")
        # ax.set_xlim(0, 1)
        ax.set_ylabel('Y', fontsize='20', fontweight="bold")
        # ax.set_ylim(0, 1)
        ax.set_zlabel('Z', fontsize='20', fontweight="bold")
        # ax.set_zlim(0, 1)

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_suffix',
        type=str,
        default="",
        help='which generated dataset to use',
    )
    args = parser.parse_args()
    

    # load the pointcloud pairs
    # data = np.load("../../kinect_processor/scenes/0/segmented_objects.npz")
    data = np.load("../data/train/left_right_0.npz")
    print(data.files)
    augmenter = SpatialAugmenter(verbose=True)

    n = 3
    branch_0 = data['branch_0'][:n]
    branch_1 = data['branch_1'][:n]
    branch_0_aug, branch_1_aug = augmenter.augment(branch_0, branch_1)

    for i in range(n):
        augmenter.plot([branch_0[i], branch_1[i]], [branch_0_aug[i], branch_1_aug[i]])