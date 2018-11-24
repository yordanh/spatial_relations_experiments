'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

data = np.load("rosbag_dump/rosbag_dump.npz")
print(data.files)

clouds = data['arr_0']
print(len(clouds))
for cloud in clouds:
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	for color in cloud:
		points = cloud[color]

		xs = points[:,0][::3]
		ys = points[:,1][::3]
		zs = points[:,2][::3]

		xs_filtered = list(filter(lambda point : (point != [0,0,0]).all(), xs))
		ys_filtered = list(filter(lambda point : (point != [0,0,0]).all(), ys))
		zs_filtered = list(filter(lambda point : (point != [0,0,0]).all(), zs))

		ax.scatter(xs_filtered, ys_filtered, zs_filtered, c=color)

	ax.set_xlabel('Z0', fontsize='20', fontweight="bold")
	# ax.set_xlim(0, 1)
	ax.set_ylabel('Z1', fontsize='20', fontweight="bold")
	# ax.set_ylim(0, 1)
	ax.set_zlabel('Z2', fontsize='20', fontweight="bold")
	# ax.set_zlim(0, 1)

	plt.show()
