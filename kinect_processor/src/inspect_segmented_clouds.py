#!/usr/bin/env python
"""
title           :inspect_segmented_clouds.py
description     :Visualises the segmented objects from each point cloud under
				:rosbag_dumps/segmented_objects.npz
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :11/2018
python_version  :2.7.6
==============================================================================
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

data = np.load("scenes/0/segmented_objects.npz")
print(data.files)

clouds = data['arr_0']
print(len(clouds))
for cloud in clouds[::5]:
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	for color in cloud:
		points = cloud[color].reshape(200*200,3)
		filtered_points = np.array(list(filter(lambda row : filter(lambda point : (point != [0,0,0]).all(), row), points)))

		xs = filtered_points[...,0][::3]
		ys = filtered_points[...,1][::3]
		zs = filtered_points[...,2][::3]

		# xs = points[:,0][::3]
		# ys = points[:,1][::3]
		# zs = points[:,2][::3]

		# points_filtered = list((lambda row: list(filter(lambda point : (point != [0,0,0]).all(), row)), points))
		# print(points_filtered)

		# xs = points[...,0][::5]
		# ys = points[...,1][::5]
		# zs = points[...,2][::5]

		# xs_filtered = list(filter(lambda point : (point != [0]), xs))
		# ys_filtered = list(filter(lambda point : (point != [0]), ys))
		# zs_filtered = list(filter(lambda point : (point != [0]), zs))

		ax.scatter(xs, ys, zs, c=color.split('_')[0])

	ax.set_xlabel('Z0', fontsize='20', fontweight="bold")
	# ax.set_xlim(0, 1)
	ax.set_ylabel('Z1', fontsize='20', fontweight="bold")
	# ax.set_ylim(0, 1)
	ax.set_zlabel('Z2', fontsize='20', fontweight="bold")
	# ax.set_zlim(0, 1)

	plt.show()
