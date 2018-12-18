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

import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Annotate the data for a particular scene')
parser.add_argument('--scene', '-sc', default='0',
                    help='Index for a scene/setup')
parser.add_argument('--every_nth', default=5, type=int, 
                    help='Every nth datapoint inspected')
args = parser.parse_args()

data = np.load('scenes/' + args.scene + '/segmented_objects.npz')
cmap = plt.cm.get_cmap('cool')

clouds = data['arr_0']
print(len(clouds))
for i, cloud in enumerate(clouds[::args.every_nth]):

	print("{0}/{1}, {2} objects".format(args.every_nth * i, len(clouds), len(cloud)))
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	for j, label in enumerate(cloud.keys()):
		points = cloud[label].reshape(200*200,3)
		filtered_points = np.array(list(filter(lambda row : filter(lambda point : (point != [0,0,0]).all(), row), points)))

		xs = filtered_points[...,0][::3]
		ys = filtered_points[...,1][::3]
		zs = filtered_points[...,2][::3]

		# ax.scatter(xs, ys, zs, c=label.split('_')[0])
		ax.scatter(xs, ys, zs, c=cmap(j / float(len(cloud.keys()))))

	ax.set_xlabel('X', fontsize='20', fontweight="bold")
	# ax.set_xlim(0, 1)
	ax.set_ylabel('Y', fontsize='20', fontweight="bold")
	# ax.set_ylim(0, 1)
	ax.set_zlabel('Z', fontsize='20', fontweight="bold")
	# ax.set_zlim(0, 1)

	plt.show()
