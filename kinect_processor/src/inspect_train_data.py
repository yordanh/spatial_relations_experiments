'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp

DATA_PATH = "../learning_experiments/data/train/"

keys = os.listdir(DATA_PATH)
keys = [x.replace('.npz', '') for x in keys]

for key in keys:
	data = np.load(osp.join(DATA_PATH, key + '.npz'))
	for i in range(0, len(data['branch_0']), 10):
		fig = plt.figure()
		fig.canvas.set_window_title(key)
		ax = fig.gca(projection='3d')

		points = data['branch_0'][i]	

		xs = points[:,0][::3]
		ys = points[:,1][::3]
		zs = points[:,2][::3]

		xs_filtered = list(filter(lambda point : (point != [0,0,0]).all(), xs))
		ys_filtered = list(filter(lambda point : (point != [0,0,0]).all(), ys))
		zs_filtered = list(filter(lambda point : (point != [0,0,0]).all(), zs))

		ax.scatter(xs_filtered, ys_filtered, zs_filtered, c='r')


		points = data['branch_1'][i]	

		xs = points[:,0][::3]
		ys = points[:,1][::3]
		zs = points[:,2][::3]

		xs_filtered = list(filter(lambda point : (point != [0,0,0]).all(), xs))
		ys_filtered = list(filter(lambda point : (point != [0,0,0]).all(), ys))
		zs_filtered = list(filter(lambda point : (point != [0,0,0]).all(), zs))

		ax.scatter(xs_filtered, ys_filtered, zs_filtered, c='c')

		ax.set_title(key.split('_')[data['label'][i]])
		ax.set_xlabel('Z0', fontsize='20', fontweight="bold")
		# ax.set_xlim(0, 1)
		ax.set_ylabel('Z1', fontsize='20', fontweight="bold")
		# ax.set_ylim(0, 1)
		ax.set_zlabel('Z2', fontsize='20', fontweight="bold")
		# ax.set_zlim(0, 1)

		plt.show()
