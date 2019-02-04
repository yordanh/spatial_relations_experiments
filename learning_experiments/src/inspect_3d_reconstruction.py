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
import cv2
import os

parser = argparse.ArgumentParser(description='Chainer example: VAE')
parser.add_argument('--out', '-o', default='result/reconstruction_arrays/',
					help='Directory to output the result')
parser.add_argument('--type', '-t', default='train',
					help='Determines whether trainign or testing reconstructions are to be inspected')

args = parser.parse_args()


def inspect_3d(args):
	no_images = 1
	no_plots = 10
	data = np.load(os.path.join(args.out, args.type + '.npz'))
	n = len(data['rec_b0'])
	print(n)

	for k in range(0, n, n/no_plots):
		fig = plt.figure()
		for i in range(no_images):
			
			for j in range(no_images):

				ax = fig.add_subplot(no_images, no_images, i * no_images + j + 1, projection='3d')
				ax.set_title("Gt/Rec " + str(i + k * no_images) + ", Branch " + str(j))
				# ax.set_title("Suplot #" + str(i * no_images + j + 1))
				every_n = 3
				xs = data['gt_b' + str(j)][i + k * no_images][..., 0][::every_n]
				ys = data['gt_b' + str(j)][i + k * no_images][..., 1][::every_n]
				zs = data['gt_b' + str(j)][i + k * no_images][..., 2][::every_n]
				ax.scatter(xs, ys, zs, c='c', alpha=0.5)

				xs = data['rec_b' + str(j)][i + k * no_images][..., 0][::every_n]
				ys = data['rec_b' + str(j)][i + k * no_images][..., 1][::every_n]
				zs = data['rec_b' + str(j)][i + k * no_images][..., 2][::every_n]
				ax.scatter(xs, ys, zs, c='r', alpha=0.1)

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

def inspect_grayscale(args):
	data = np.load(os.path.join(args.out, args.type + '.npz'))
	n = len(data['rec_b0'])
	no_plots = 10
	scale = 2.5
	dtypes = ['gt_b', 'rec_b']
	axes = ['X', 'Y', 'Z']
	branches = [0, 1]

	for k in range(0,n, n/no_plots):
		fig, ax = plt.subplots(4,4,figsize=(16, 16), dpi=75)
		
		for branch in range(len(branches)):
			for dtype in range(len(dtypes)):
				for axis in range(len(axes)):

					points = data[dtypes[dtype] + str(branch)][k][..., axis]
					points = cv2.resize(points,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

					ax[dtype + branch * 2, axis].set_title(axes[axis] + " " + dtypes[dtype].split('_')[0].upper() + " MU: {0}, STD: {1}".\
												format(round(np.mean(points[points != 0]), 2), round(np.std(points[points != 0]), 2)))
					ax[dtype + branch * 2, axis].set_xticks([])
					ax[dtype + branch * 2, axis].set_yticks([])
					ax[dtype + branch * 2, axis].imshow(points, cmap='gray')

				tmp = np.array(zip(data[dtypes[dtype] + str(branch)][k][..., 0].reshape(200*200), \
					               data[dtypes[dtype] + str(branch)][k][..., 1].reshape(200*200), \
					               data[dtypes[dtype] + str(branch)][k][..., 2].reshape(200*200))).reshape(200, 200, 3)
				tmp = cv2.resize((tmp * 255).astype(np.uint8),None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

				ax[dtype + branch * 2, len(axes)].set_title("RGB " + dtypes[dtype].split('_')[0].upper())
				ax[dtype + branch * 2, len(axes)].set_xticks([])
				ax[dtype + branch * 2, len(axes)].set_yticks([])
				ax[dtype + branch * 2, len(axes)].imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
		
		plt.show()


	# for k in range(0,n, n/no_plots):
	# 		for j in range(len(branches)):
	# 			points = data['gt_b' + str(j)][k][..., 0]
	# 			print("X GT MEAN:\t {0} \t STD:\t{1}".format(round(np.mean(points[points != 0]), 2), round(np.std(points[points != 0]), 2)))
	# 			cv2.imshow("X GT", cv2.resize(points,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC))

	# 			points = data['rec_b' + str(j)][k][..., 0]
	# 			print("X REC MEAN:\t {0} \t STD:\t{1}".format(round(np.mean(points[points != 0]), 2), round(np.std(points[points != 0]), 2)))
	# 			cv2.imshow("X REC", cv2.resize(points,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC))
				


	# 			points = data['gt_b' + str(j)][k][..., 1]
	# 			print("Y GT MEAN:\t {0} \t STD:\t{1}".format(round(np.mean(points[points != 0]), 2), round(np.std(points[points != 0]), 2)))
	# 			cv2.imshow("Y GT", cv2.resize(points,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC))

	# 			points = data['rec_b' + str(j)][k][..., 1]
	# 			print("Y REC MEAN:\t {0} \t STD:\t{1}".format(round(np.mean(points[points != 0]), 2), round(np.std(points[points != 0]), 2)))
	# 			cv2.imshow("Y REC", cv2.resize(points,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC))



	# 			points = data['gt_b' + str(j)][k][..., 2]
	# 			print("Z GT MEAN:\t {0} \t STD:\t{1}".format(round(np.mean(points[points != 0]), 2), round(np.std(points[points != 0]), 2)))
	# 			cv2.imshow("Z GT", cv2.resize(points,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC))

	# 			points = data['rec_b' + str(j)][k][..., 2]
	# 			print("Z REC MEAN:\t {0} \t STD:\t{1}".format(round(np.mean(points[points != 0]), 2), round(np.std(points[points != 0]), 2)))
	# 			cv2.imshow("Z REC", cv2.resize(points,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC))



	# 			tmp = np.array(zip(data['gt_b' + str(j)][k][..., 0].reshape(200*200), data['gt_b' + str(j)][k][..., 1].reshape(200*200), data['gt_b' + str(j)][k][..., 2].reshape(200*200))).reshape(200, 200, 3)
	# 			cv2.imshow("RGB GT", cv2.resize((tmp * 255).astype(np.uint8),None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC))

	# 			tmp = np.array(zip(data['rec_b' + str(j)][k][..., 0].reshape(200*200), data['rec_b' + str(j)][k][..., 1].reshape(200*200), data['rec_b' + str(j)][k][..., 2].reshape(200*200))).reshape(200, 200, 3)
	# 			cv2.imshow("RGB REC", cv2.resize((tmp * 255).astype(np.uint8),None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC))

	# 			print("\n")

	# 			cv2.waitKey(0)

if __name__ == "__main__":
	inspect_3d(args)
	# inspect_grayscale(args)