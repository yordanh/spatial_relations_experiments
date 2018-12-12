import argparse
import os
import os.path as osp
import cv2
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib
# matplotlib.use('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import subprocess
import shutil

import chainer
from chainer import training
from chainer.training import extensions
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu
import chainer.functions as F
from chainer import serializers

import net_200x200 as net
import data_generator
from config_parser import ConfigParser
from utils import *


def save_reconstruction_arrays(data, model, folder_name="."):

	print("Clear Images from Last Reconstructions\n")
	all_files = list(filter(lambda filename : '.' in filename, os.listdir(folder_name)))
	map(lambda x : os.remove(folder_name + x), all_files)
	print("Saving Array RECONSTRUCTIONS\n")

	(train_b0, train_b1) = data

	no_images = 10
	train_ind = np.linspace(0, len(train_b0) - 1, no_images, dtype=int)
	result = model(train_b0[train_ind], train_b1[train_ind])

	gt_b0 = np.swapaxes(train_b0[train_ind], 1, 3)
	gt_b1 = np.swapaxes(train_b1[train_ind], 1, 3)

	rec_b0 = np.swapaxes(result[0].data, 1, 3)
	rec_b1 = np.swapaxes(result[1].data, 1, 3)

	output = {"gt_b0": gt_b0, "gt_b1": gt_b1, 'rec_b0': rec_b0, 'rec_b1': rec_b1}
	np.savez(os.path.join("result", "reconstruction_arrays/train" + ".npz"), **output)


def eval_seen_data(data, model, groups, folder_name="."):

	print("Clear Images from Last Seen Scatter\n")
	all_files = list(filter(lambda filename : '.' in filename, os.listdir(folder_name)))
	map(lambda x : os.remove(folder_name + x), all_files)
	print("Evaluating on SEEN data\n")

	(data_b0, data_b1) = data
	n = 100
	every_nth = len(data_b0) / n
	if every_nth == 0:
		every_nth = 1

	axis_ranges = [-5, 5]
	for group_key in groups:
		for label in groups[group_key]:

			print("Visualising label:\t{0}, Group:\t{1}".format(label, group_key))

			indecies = [i for i, x in enumerate(train_labels) if x == label]
			filtered_data_b0 = data_b0.take(indecies, axis=0)[::every_nth]
			filtered_data_b1 = data_b1.take(indecies, axis=0)[::every_nth]

			latent_mu = model.get_latent(filtered_data_b0, filtered_data_b1).data
			pairs = [(0,1), (0,2), (1,2)]
			for pair in pairs:
				plt.scatter(latent_mu[:, pair[0]], latent_mu[:, pair[1]], c='red', label=label, alpha=0.75)
				plt.grid()

				# major axes
				plt.plot([axis_ranges[0], axis_ranges[1]], [0,0], 'k')
				plt.plot([0,0], [axis_ranges[0], axis_ranges[1]], 'k')

				plt.xlim(axis_ranges[0], axis_ranges[1])
				plt.ylim(axis_ranges[0], axis_ranges[1])

				plt.xlabel("Z_" + str(pair[0]))
				plt.ylabel("Z_" + str(pair[1]))

				plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
				plt.savefig(osp.join(folder_name, "group_" + str(group_key) + "_" + label + "_Z_" + str(pair[0]) + "_Z_" + str(pair[1])), bbox_inches="tight")
				plt.close()


def eval_seen_data_single(data, model, labels=[], folder_name="."):
	
	print("Clear Images from Last Seen Scatter Single\n")
	all_files = list(filter(lambda filename : '.' in filename, os.listdir(folder_name)))
	map(lambda x : os.remove(folder_name + x), all_files)
	print("Evaluating on SEEN SINGLE data\n")

	(data_b0, data_b1) = data

	axis_ranges = [-5, 5]
	pairs = [(0,1), (0,2), (1,2)]
	n = 100
	every_nth = len(data_b0) / n
	if every_nth == 0:
		every_nth = 1

	filtered_data_b0 = data_b0.take(range(len(data_b0)), axis=0)[::every_nth]
	filtered_data_b1 = data_b1.take(range(len(data_b1)), axis=0)[::every_nth]
	labels = labels[::every_nth]

	latent_mu = model.get_latent(filtered_data_b0, filtered_data_b1).data

	filtered_data_b0 = np.swapaxes(filtered_data_b0, 1, 3)
	filtered_data_b1 = np.swapaxes(filtered_data_b1, 1, 3)

	for i in range(0, len(latent_mu), 30):

		fig = plt.figure()
		fig.canvas.set_window_title(labels[i])

		ax = fig.add_subplot(1, 4, 1, projection='3d')
		points = filtered_data_b0[i].reshape(200*200,3)
		filtered_points = np.array(list(filter(lambda row : filter(lambda point : (point != [0,0,0]).all(), row), points)))
		xs_0 = filtered_points[...,0][::3]
		ys_0 = filtered_points[...,1][::3]
		zs_0 = filtered_points[...,2][::3]
		ax.scatter(xs_0, ys_0, zs_0, c='r', alpha=0.5)

		points = filtered_data_b1[i].reshape(200*200,3)
		filtered_points = np.array(list(filter(lambda row : filter(lambda point : (point != [0,0,0]).all(), row), points)))
		xs_1 = filtered_points[...,0][::3]
		ys_1 = filtered_points[...,1][::3]
		zs_1 = filtered_points[...,2][::3]
		ax.scatter(xs_1, ys_1, zs_1, c='c', alpha=0.5)

		ax.set_xlabel('X', fontweight="bold")
		ax.set_ylabel('Y', fontweight="bold")
		ax.set_zlabel('Z', fontweight="bold")

		for j, pair in enumerate(pairs):
			ax = fig.add_subplot(2, 4, j + 2)
			ax.scatter(latent_mu[i, pair[0]], latent_mu[i, pair[1]], c='red', label="unseen", alpha=0.75)
			ax.grid()

			# major axes
			ax.plot([axis_ranges[0], axis_ranges[1]], [0,0], 'k')
			ax.plot([0,0], [axis_ranges[0], axis_ranges[1]], 'k')

			ax.set_xlim(axis_ranges[0], axis_ranges[1])
			ax.set_ylim(axis_ranges[0], axis_ranges[1])

			ax.set_xlabel("Z_" + str(pair[0]))
			ax.set_ylabel("Z_" + str(pair[1]))

			# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

		# plt.savefig(osp.join(folder_name, str(i) + "_Z_" + str(pair[0]) + "_Z_" + str(pair[1])), bbox_inches="tight")
		# plt.close()

		plt.show()


def eval_unseen_data(data, model, folder_name="."):
	
	print("Clear Images from Last Unseen Scatter\n")
	all_files = list(filter(lambda filename : '.' in filename, os.listdir(folder_name)))
	map(lambda x : os.remove(folder_name + x), all_files)
	print("Evaluating on UNSEEN data\n")

	(data_b0, data_b1) = data

	axis_ranges = [-5, 5]
	pairs = [(0,1), (0,2), (1,2)]
	n = 100
	every_nth = len(data_b0) / n
	if every_nth == 0:
		every_nth = 1

	filtered_data_b0 = data_b0.take(range(len(data_b0)), axis=0)[::every_nth]
	filtered_data_b1 = data_b1.take(range(len(data_b1)), axis=0)[::every_nth]

	latent_mu = model.get_latent(filtered_data_b0, filtered_data_b1).data
	latent_mu_flipped = model.get_latent(filtered_data_b1, filtered_data_b0).data

	filtered_data_b0 = np.swapaxes(filtered_data_b0, 1, 3)
	filtered_data_b1 = np.swapaxes(filtered_data_b1, 1, 3)

	for i in range(0, len(latent_mu), 5):

		fig = plt.figure()

		ax = fig.add_subplot(2, 4, 1, projection='3d')
		points = filtered_data_b0[i].reshape(200*200,3)
		filtered_points = np.array(list(filter(lambda row : filter(lambda point : (point != [0,0,0]).all(), row), points)))
		xs_0 = filtered_points[...,0][::3]
		ys_0 = filtered_points[...,1][::3]
		zs_0 = filtered_points[...,2][::3]
		ax.scatter(xs_0, ys_0, zs_0, c='r', alpha=0.5)

		points = filtered_data_b1[i].reshape(200*200,3)
		filtered_points = np.array(list(filter(lambda row : filter(lambda point : (point != [0,0,0]).all(), row), points)))
		xs_1 = filtered_points[...,0][::3]
		ys_1 = filtered_points[...,1][::3]
		zs_1 = filtered_points[...,2][::3]
		ax.scatter(xs_1, ys_1, zs_1, c='c', alpha=0.5)

		ax.set_xlabel('X', fontweight="bold")
		ax.set_ylabel('Y', fontweight="bold")
		ax.set_zlabel('Z', fontweight="bold")

		for j, pair in enumerate(pairs):
			ax = fig.add_subplot(2, 4, j + 2)
			ax.scatter(latent_mu[i, pair[0]], latent_mu[i, pair[1]], c='red', label="unseen", alpha=0.75)
			ax.grid()

			# major axes
			ax.plot([axis_ranges[0], axis_ranges[1]], [0,0], 'k')
			ax.plot([0,0], [axis_ranges[0], axis_ranges[1]], 'k')

			ax.set_xlim(axis_ranges[0], axis_ranges[1])
			ax.set_ylim(axis_ranges[0], axis_ranges[1])

			ax.set_xlabel("Z_" + str(pair[0]))
			ax.set_ylabel("Z_" + str(pair[1]))

			# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)



		ax = fig.add_subplot(2, 4, 5, projection='3d')
		ax.scatter(xs_1, ys_1, zs_1, c='r', alpha=0.5)
		ax.scatter(xs_0, ys_0, zs_0, c='c', alpha=0.5)

		ax.set_xlabel('X', fontweight="bold")
		ax.set_ylabel('Y', fontweight="bold")
		ax.set_zlabel('Z', fontweight="bold")

		for j, pair in enumerate(pairs):
			ax = fig.add_subplot(2, 4, j + 6)
			ax.scatter(latent_mu_flipped[i, pair[0]], latent_mu_flipped[i, pair[1]], c='red', label="unseen", alpha=0.75)
			ax.grid()

			# major axes
			ax.plot([axis_ranges[0], axis_ranges[1]], [0,0], 'k')
			ax.plot([0,0], [axis_ranges[0], axis_ranges[1]], 'k')

			ax.set_xlim(axis_ranges[0], axis_ranges[1])
			ax.set_ylim(axis_ranges[0], axis_ranges[1])

			ax.set_xlabel("Z_" + str(pair[0]))
			ax.set_ylabel("Z_" + str(pair[1]))

			# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
		
		# plt.savefig(osp.join(folder_name, str(i) + "_Z_" + str(pair[0]) + "_Z_" + str(pair[1])), bbox_inches="tight")
		# plt.close()

		plt.show()

if __name__ == "__main__":

	ignore = ["unlabelled", "train"]
	generator = data_generator.DataGenerator()
	train_b0, train_b1, train_labels, train_concat, train_vectors, test_b0, test_b1, test_labels, test_concat, test_vectors, unseen_b0, unseen_b1,\
	unseen_labels, groups = generator.generate_dataset(ignore=ignore, args=None)

	print('\n###############################################')
	print("DATA_LOADED")
	print("# Training Branch 0: \t\t{0}".format(train_b0.shape))
	print("# Training Branch 1: \t\t{0}".format(train_b1.shape))
	print("# Training labels: \t{0}".format(set(train_labels)))
	print("# Training labels: \t{0}".format(train_labels.shape))
	print("# Training concat: \t{0}".format(len(train_concat)))
	print("# Training vectors: \t{0}".format(train_vectors.shape))
	print("# Testing Branch 0: \t\t{0}".format(test_b0.shape))
	print("# Testing Branch 1: \t\t{0}".format(test_b1.shape))
	print("# Testing labels: \t{0}".format(set(test_labels)))
	print("# Testing concat: \t{0}".format(len(test_concat)))
	print("# Testing labels: \t{0}".format(test_labels.shape))
	print("# Testing vectors: \t{0}".format(test_vectors.shape))
	print("# Unseen Branch 0: \t\t{0}".format(unseen_b0.shape))
	print("# Unseen Branch 1: \t\t{0}".format(unseen_b1.shape))
	print("# Unseen labels: \t{0}".format(set(unseen_labels)))

	print("\n# Groups: \t{0}".format(groups))
	print('###############################################\n')

	model = net.Conv_Siam_VAE(train_b0.shape[1], train_b1.shape[1], n_latent=8, groups=groups, alpha=1, beta=1, gamma=1)
	serializers.load_npz("result/models/final.model", model)
	model.to_cpu()	

	# save the pointcloud reconstructions
	# save_reconstruction_arrays((train_b0, train_b0), model, folder_name="result/reconstruction_arrays/")


	# evaluate on the data that was seen during trainig
	# eval_seen_data((train_b0, train_b1), model, groups, folder_name="result/scatter/seen/")

	# evaluate on the data that was seen during trainig one by one + 3D
	# eval_seen_data_single((test_b0, test_b1), model, labels=test_labels, folder_name="result/scatter/seen_single/")

	# evaluate on the data that was NOT seen during trainig
	eval_unseen_data((unseen_b0, unseen_b1), model, folder_name="result/scatter/unseen/")