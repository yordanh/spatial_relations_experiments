'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
'''

import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp

parser = argparse.ArgumentParser(description='Annotate the data for a particular scene')
parser.add_argument('--dtype', default='train', type=str,
                    help='Type of data to inspect')
parser.add_argument('--every_nth', default=10, type=int, 
                    help='Every nth datapoint inspected')
args = parser.parse_args()

DATA_PATH = osp.join("../learning_experiments/data", args.dtype)

keys = os.listdir(DATA_PATH)
keys = [x.replace('.npz', '') for x in keys]

for key in keys:
	data = np.load(osp.join(DATA_PATH, key + '.npz'))
	for i in range(0, len(data['branch_0']), args.every_nth):

		print("{0}/{1}".format(i, len(data['branch_0'])))
		fig = plt.figure()
		fig.canvas.set_window_title(key)
		ax = fig.gca(projection='3d')

		points = data['branch_0'][i].reshape(200*200,3)
		filtered_points = np.array(list(filter(lambda row : filter(lambda point : (point != [0,0,0]).all(), row), points)))

		xs = filtered_points[...,0][::3]
		ys = filtered_points[...,1][::3]
		zs = filtered_points[...,2][::3]

		ax.scatter(xs, ys, zs, c='r')



		points = data['branch_1'][i].reshape(200*200,3)
		filtered_points = np.array(list(filter(lambda row : filter(lambda point : (point != [0,0,0]).all(), row), points)))

		xs = filtered_points[...,0][::3]
		ys = filtered_points[...,1][::3]
		zs = filtered_points[...,2][::3]

		ax.scatter(xs, ys, zs, c='c')

		ax.set_title(key.split('_')[data['label'][i]])
		ax.set_xlabel('X', fontsize='20', fontweight="bold")
		# ax.set_xlim(0, 1)
		ax.set_ylabel('Y', fontsize='20', fontweight="bold")
		# ax.set_ylim(0, 1)
		ax.set_zlabel('Z', fontsize='20', fontweight="bold")
		# ax.set_zlim(0, 1)

		plt.show()
