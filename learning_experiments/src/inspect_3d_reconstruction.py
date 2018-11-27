#!/usr/bin/env python
"""
title           :inspect_3d_reconstruction.py
description     :Plots original XYZ images and the corresponding reconstructions.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :10/2018
python_version  :2.7.6
==============================================================================
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Chainer example: VAE')
parser.add_argument('--out', '-o', default='result/reconstruction_arrays/',
					help='Directory to output the result')
parser.add_argument('--type', '-t', default='train',
					help='Determines whether trainign or testing reconstructions are to be inspected')

args = parser.parse_args()

no_images = 2
no_plots = 3
data = np.load(os.path.join(args.out, args.type + '.npz'))

for k in range(no_plots):
	fig = plt.figure()
	for i in range(no_images):
		
		for j in range(2):

			ax = fig.add_subplot(no_images, no_images, i * no_images + j + 1, projection='3d')
			ax.set_title("Gt/Rec " + str(i + k * 2) + ", Branch " + str(j))
			# ax.set_title("Suplot #" + str(i * no_images + j + 1))
			every_n = 3
			xs = data['gt_b' + str(j)][i + k * 2][..., 0][::every_n]
			ys = data['gt_b' + str(j)][i + k * 2][..., 1][::every_n]
			zs = data['gt_b' + str(j)][i + k * 2][..., 2][::every_n]
			ax.scatter(xs, ys, zs, c='c', alpha=0.5)

			xs = data['rec_b' + str(j)][i + k * 2][..., 0][::every_n]
			ys = data['rec_b' + str(j)][i + k * 2][..., 1][::every_n]
			zs = data['rec_b' + str(j)][i + k * 2][..., 2][::every_n]
			ax.scatter(xs, ys, zs, c='r', alpha=0.5)

			ax.xaxis.set_major_locator(plt.MaxNLocator(2))
			ax.yaxis.set_major_locator(plt.MaxNLocator(2))
			ax.zaxis.set_major_locator(plt.MaxNLocator(2))

			ax.set_xlabel('Z0/X', fontweight="bold")
			# ax.set_xlim(0, 1)
			ax.set_ylabel('Z1/Y', fontweight="bold")
			# ax.set_ylim(0, 1)
			ax.set_zlabel('Z2/Z', fontweight="bold")
			# ax.set_zlim(0, 1)

plt.show()
