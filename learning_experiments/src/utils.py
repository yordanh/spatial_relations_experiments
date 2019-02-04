#!/usr/bin/env python
"""
title           :utils.py
description     :Utility functions to be used for result processing after the model training phase.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :11/2018
python_version  :2.7.14
==============================================================================
"""

import os
import os.path as osp
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import entropy
import cv2
from scipy.stats import norm
from math import sqrt
import itertools
import  copy
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('agg')
import matplotlib.pyplot as plt
import shutil

import chainer
import chainer.functions as F

########################################
############ UTIL FUNCTIONS ############
########################################

def plot_xyz(xyz_points, args=None):

    xs = xyz_points[:,:,0][::5]
    ys = xyz_points[:,:,1][::5]
    zs = xyz_points[:,:,2][::5]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(xs, ys, zs, c='c')
    
    ax.set_xlabel('X', fontsize='20', fontweight="bold")
    ax.set_xlim(-1, 1)
    ax.set_ylabel('Y', fontsize='20', fontweight="bold")
    ax.set_ylim(-1, 1)
    ax.set_zlabel('Z', fontsize='20', fontweight="bold")
    ax.set_zlim(0, 1)

    plt.savefig(os.path.join(args.out + "3dscatter"), bbox_inches="tight")
    plt.close()


# delete all result files from the output folder
def clear_last_results(folder_name=None):
    all_files = list(filter(lambda filename : '.' in filename, os.listdir(folder_name)))
    map(lambda x : os.remove(folder_name + x), all_files)

    # subfolders = ["scatter", "reconstruction_arrays"]
    # for subfolder in subfolders:
	   #  map(lambda x : os.remove(osp.join(folder_name, subfolder, x)), os.listdir(osp.join(folder_name, subfolder)))

    # leftover_folders = list(filter(lambda filename : filename != "models", os.listdir(folder_name)))
    # map(lambda x : shutil.rmtree(folder_name + x), leftover_folders)

    # os.mkdir(folder_name + "gifs")
    # os.mkdir(folder_name + "gifs/manifold_gif")
    # os.mkdir(folder_name + "gifs/scatter_gif")
    # os.mkdir(folder_name + "scatter")
    # os.mkdir(folder_name + "eval")
    # os.mkdir(folder_name + "reconstruction_arrays")


# for a given set of example images, calculate their reconstructions
def perform_reconstructions(model=None, train=None, test=None, unseen=None, no_images=None, name_suffix=None, args=None):
    train_ind = np.linspace(0, len(train) - 1, no_images, dtype=int)
    x = chainer.Variable(np.asarray(train[train_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = model(x)
        z1 = model.get_latent_mu(x)
    save_images(x=x.data, z=[], no_images=no_images, filename=os.path.join(args.out, 'train_' + name_suffix), args=args)
    save_images(x=x1.data,z=z1.data, no_images=no_images, filename=os.path.join(args.out, 'train_' + name_suffix + "_rec"), 
                args=args)

    # reconstruct testing examples
    test_ind = np.linspace(0, len(test) - 1, no_images, dtype=int)
    x = chainer.Variable(np.asarray(test[test_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = model(x)
        z1 = model.get_latent_mu(x)
    save_images(x=x.data, z=[], no_images=no_images, filename=os.path.join(args.out, 'test_' + name_suffix), args=args)
    save_images(x=x1.data,z=z1.data, no_images=no_images, filename=os.path.join(args.out, 'test_' + name_suffix + "_rec"), 
                args=args)

    # reconstruct unseen examples
    if len(unseen) != 0:
        unseen_ind = np.linspace(0, len(unseen) - 1, no_images, dtype=int)
        x = chainer.Variable(np.asarray(unseen[unseen_ind]))
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1 = model(x)
            z1 = model.get_latent_mu(x)
        save_images(x=x.data, z=[], no_images=no_images, filename=os.path.join(args.out, 'unseen_' + name_suffix), args=args)
        save_images(x=x1.data,z=z1.data, no_images=no_images, filename=os.path.join(args.out, 'unseen_' + name_suffix + "_rec"), 
                    args=args)

    # # draw images from randomly sampled z under a 'vanilla' normal distribution
    # z = chainer.Variable(
    #     np.random.normal(0, 1, (no_images, args.dimz)).astype(np.float32))
    # x = model.decode(z)
    # save_images(x=x.data, z=z.data, no_images=no_images, filename=os.path.join(args.out, 'sampled_' + name_suffix), 
    #             args=args)


# plot and save loss and accuracy curves
def plot_loss_curves(stats=None, args=None):
    # overall train/validation losses
    plt.figure(figsize=(10, 10))
    plt.grid()
    colors = ['r', 'k', 'b', 'g', 'gold', 'cyan', 'magenta', 'brown', 'aqua', 'olive']
    for i, channel in enumerate(stats):
        if channel == "valid_label_loss" or channel == "valid_label_acc" or channel == "train_accs" or channel == "valid_label_acc_1" or channel == "valid_label_acc_2":
            continue
        plt.plot(range(len(stats[channel])),stats[channel], color=colors[i], label=channel)
    plt.xlabel("Epoch #", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)
    plt.savefig(os.path.join(args.out, "losses"), bbox_inches="tight")
    plt.close()

    # validation label loss
    plt.figure(figsize=(10, 10))
    plt.grid()
    plt.plot(range(len(stats['valid_label_loss'])),stats['valid_label_loss'], color='g', label='valid_label_loss')
    plt.xlabel("Epoch #", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)
    plt.savefig(os.path.join(args.out, "label_loss"), bbox_inches="tight")
    plt.close()

    # validation label accuracy
    plt.figure(figsize=(10, 10))
    plt.grid()
    plt.plot(range(len(stats['valid_label_acc'])),stats['valid_label_acc'], color='r', label='valid_label_acc')
    plt.xlabel("Epoch #", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)
    plt.savefig(os.path.join(args.out, "valid_label_acc"), bbox_inches="tight")
    plt.close()

    # training label accuracy
    plt.figure(figsize=(10, 10))
    plt.grid()
    plt.plot(range(len(stats['train_accs'])),stats['train_accs'], color='b', label='train_accs')
    plt.xlabel("Epoch #", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)
    plt.savefig(os.path.join(args.out + "train_label_acc"), bbox_inches="tight")
    plt.close()


# calculate statistics for the predicted labels
def compare_labels(test=None, test_labels=None, model=None, args=None, cuttoff_thresh=1):

    mu, ln_var = model.encode(test)
    
    if args.labels == "composite":
        hat_labels_0, hat_labels_1 = model.predict_label(mu, ln_var, softmax=True)

        print("\nValidation Accuracy for group 0: {0}\n".format(F.accuracy(hat_labels_0, test_labels[0])))
        print("\nValidation Accuracy for group 1: {0}\n".format(F.accuracy(hat_labels_1, test_labels[1])))
    elif args.labels == "singular":
        hat_labels = model.predict_label(mu, ln_var, softmax=True)

        print("Validation Accuracy: {0}\n".format(F.accuracy(hat_labels, test_labels)))


def save_intermediate(data=None, labels=None, model=None, args=None):
    
    counters = {key: 0 for key in set(labels)}
    batch_counter = 0
    print(len(data))

    while batch_counter * args.batchsize <= len(data):
        print(counters)
       
        start = batch_counter * args.batchsize
        end = start + args.batchsize
        if len(data[start:end]) == 0:
            break 
        data_inter = model(data[start:end]).data
        labels_tmp = labels[start:end]

        for x, label in zip(data_inter, labels_tmp):
            x = np.swapaxes(x, 0, 2)
            x = np.array(x*255, dtype=np.uint8)

            cv2.imwrite(os.path.join("data", "dSprites", "train", label ,str(counters[label])) + ".png", x)
            counters[label] += 1

        batch_counter += 1

    if batch_counter * args.batchsize != len(data):
        start = (batch_counter - 1) * args.batchsize
        end = start + args.batchsize 
        data_inter = model(data[start:end]).data
        labels_tmp = labels[start:end]

        for x, label in zip(data_inter, labels_tmp):
            print(label)
            x = np.swapaxes(x, 0, 2)
            x = np.array(x*255, dtype=np.uint8)

            cv2.imwrite(os.path.join("data", "dSprites", "train", label ,str(counters[label])) + ".png", x)
            counters[label] += 1

    print(counters)



# visualize the results
def save_images(x=None, z=None, no_images=None, filename=None, args=None):

    fig, ax = plt.subplots(int(sqrt(no_images)), int(sqrt(no_images)), figsize=(9, 9), dpi=100)
    for i, (ai, xi) in enumerate(zip(ax.flatten(), x)):
        
        if len(z) != 0:
            zi = z[i]
        else:
            zi=None
        if args.model == "conv":
            xi = np.swapaxes(xi, 0, 2)
        else:
            if args.data == "mnist":
                xi = xi.reshape(28, 28)
            else:
                xi = xi.reshape(100, 100, 3)
        
        if xi.shape[-1] == 1:
            xi = xi.reshape(xi.shape[:-1])
        if zi is not None:
            ai.set_title("{0}; {1}".format(round(zi[0],2), round(zi[1],2)))

        ai.set_xticks([])
        ai.set_yticks([])
        image = ai.imshow(cv2.cvtColor(xi, cv2.COLOR_BGR2RGB))
    fig.savefig(filename)
    plt.close()


# attach a color to each singular and composite class labels, both for their data points 
# and fitted overlayed distributions
def attach_colors(labels=None, n_groups=None, composite=True):

    colors = ['c', 'b', 'g', 'y', 'k', 'orange', 'maroon', 'lime', 'salmon', 
              'crimson', 'gold', 'coral', 'navy', 'purple', 'olive', 'r', 'yellowgreen', 'brown', 'indigo', 'teal', 'turquoise',
		'r','g','b','orange','y','c','k']
    result = {"singular":{}, "composite":{}}

    counter = 0
    for label in sorted(set(labels)):
        if label in result["singular"]:
            continue
        else:
            result["singular"][label] = {}
            result["singular"][label]["data"] = colors[counter]
            result["singular"][label]["dist"] = colors[counter]
            counter += 1

    if "unknown" not in result["singular"].keys():
        result["singular"]["unknown"] = {}
        result["singular"]["unknown"]["dist"] = colors[-1]
    if composite:
        if n_groups > 1:
            labels = labels.reshape(len(labels) / n_groups, n_groups)
            labels = np.array(["_".join(x) for x in labels])

        counter = 0
        for label in sorted(set(labels)):
            if label in result["composite"]:
                continue
            else:
                result["composite"][label] = {}
                result["composite"][label]["data"] = colors[counter]
                result["composite"][label]["dist"] = colors[counter]
                counter += 1

    return result


# plot a set of input datapoitns to the latent space and fit a normal distribution over the projections
# show the contours for the overall data distribution
def plot_overall_distribution(data=None, labels=None, groups=None, boundaries=None, colors=None, model=None, 
                              overlay=True, spread=True, filename=None, mode=None):
    latent_all = None
    if mode == "singular":
        labels = np.array([labels[i::len(groups)] for i in range(len(groups))])
    elif mode == "composite":
        labels = labels.reshape(len(labels) / len(groups), len(groups))
        labels = np.array([["_".join(x) for x in labels]])

    # scatter plot all the data points in the latent space
    plt.figure(figsize=(10, 10))
    for concept_group_labels in labels:
        for label in sorted(set(concept_group_labels)):
            indecies = [i for i, x in enumerate(concept_group_labels) if x == label]
            filtered_data = chainer.Variable(data.take(indecies, axis=0))
            
            if spread:
                latent = model.get_latent(filtered_data).data[:,:2]
            else:
                latent = model.get_latent_mu(filtered_data).data[:,:2]
            
            plt.scatter(latent[:, 0], latent[:, 1], c=colors[label]["data"], label=str(label), alpha=0.75)

            if latent_all is not None:
                latent_all = np.append(latent_all, latent, axis=0)
            else:
                latent_all = latent
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

    # plot bounding box for the visualised manifold
    # boundaries are [[min_x, min_y],[max_x, max_y]]
    plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[0,1], boundaries[0,1]], 'r') # top line
    plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[1,1], boundaries[1,1]], 'r') # bottom line
    plt.plot([boundaries[0,0], boundaries[0,0]], [boundaries[0,1], boundaries[1,1]], 'r') # left line
    plt.plot([boundaries[1,0], boundaries[1,0]], [boundaries[0,1], boundaries[1,1]], 'r') # right line
    # major axes
    plt.plot([boundaries[0,0], boundaries[1,0]], [0,0], 'k')
    plt.plot([0,0], [boundaries[0,1], boundaries[1,1]], 'k')

    plt.grid()
    # plt.savefig(filename, bbox_inches="tight")
    
    if overlay:
        # fit and plot a distribution over all the latent projections
        delta = 0.025
        mean = np.mean(latent_all, axis=0)
        cov = np.cov(latent_all.T)
        x = np.arange(min(latent_all[:, 0]), max(latent_all[:, 0]), delta)
        y = np.arange(min(latent_all[:, 1]), max(latent_all[:, 1]), delta)
        X, Y = np.meshgrid(x, y)
        Z = multivariate_normal.pdf(np.array([zip(c,d) for c,d in zip(X,Y)]), mean=mean, cov=cov)
        plt.contour(X, Y, Z, colors='r')
        plt.title("mu[0]:{0}; mu[1]:{1}\ncov[0,0]:{2}; cov[1,1]:{3}\ncov[0,1]:{4}".format(round(mean[0],2), 
                  round(mean[1],2), round(cov[0,0],2), round(cov[1,1],2), round(cov[0,1],2)), fontweight="bold", fontsize=14)
        plt.savefig(filename + "_overlayed", bbox_inches="tight")
    plt.close()


# plot a set of input datapoitns to the latent space and fit normal distributions over the projections
# show the contours for the distribution for each label
def plot_separate_distributions(data=None, labels=None, groups=None, boundaries=None, 
                                colors=None, model=None, filename=None, overlay=True, mode=None):
    latent_all = []

    if mode == "singular":
        labels = np.array([labels[i::len(groups)] for i in range(len(groups))])
    elif mode == "composite":
        labels = labels.reshape(len(labels) / len(groups), len(groups))
        labels = np.array([["_".join(x) for x in labels]])# [[]] as if there is a single concept group

    # if we are with composite labels or with a single group there is no point is visualising that
    if len(labels) > 1:
        for key, concept_group_labels in enumerate(labels):
            plt.figure(figsize=(10, 10))
            for label in sorted(set(concept_group_labels)):
                indecies = [i for i, x in enumerate(concept_group_labels) if x == label]
                filtered_data = chainer.Variable(data.take(indecies, axis=0))
                latent = model.get_latent_mu(filtered_data)
                latent = latent.data[:,:2]

                plt.scatter(latent[:, 0], latent[:, 1], c=colors[label]["data"], label=str(label), alpha=0.75)

                if overlay:
                    delta = 0.025
                    mean = np.mean(latent, axis=0)
                    cov = np.cov(latent.T)
                    x = np.arange(min(latent[:, 0]), max(latent[:, 0]), delta)
                    y = np.arange(min(latent[:, 1]), max(latent[:, 1]), delta)
                    X, Y = np.meshgrid(x, y)
                    Z = multivariate_normal.pdf(np.array([zip(c,d) for c,d in zip(X,Y)]), mean=mean, cov=cov)
                    plt.contour(X, Y, Z, colors=colors[label]["dist"])
            
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

            # plot bounding box for the visualised manifold
            # boundaries are [[min_x, min_y],[max_x, max_y]]
            plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[0,1], boundaries[0,1]], 'r') # top line
            plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[1,1], boundaries[1,1]], 'r') # bottom line
            plt.plot([boundaries[0,0], boundaries[0,0]], [boundaries[0,1], boundaries[1,1]], 'r') # left line
            plt.plot([boundaries[1,0], boundaries[1,0]], [boundaries[0,1], boundaries[1,1]], 'r') # right line
            # major axes
            plt.plot([boundaries[0,0], boundaries[1,0]], [0,0], 'k')
            plt.plot([0,0], [boundaries[0,1], boundaries[1,1]], 'k')

            plt.grid()
            plt.savefig(filename + "_group_" + str(key) + '_overlayed', bbox_inches="tight")
            plt.close()

    # scatter plot all the data points in the latent space
    dimensions_pairs = list(itertools.combinations(range(model.n_latent), 2))
    for pair in dimensions_pairs:
        if 0 not in pair and 1 not in pair:
                continue
        plt.figure(figsize=(10, 10))
        for concept_group_labels in labels:
            for label in sorted(set(concept_group_labels)):
                indecies = [i for i, x in enumerate(concept_group_labels) if x == label]
                filtered_data = chainer.Variable(data.take(indecies, axis=0))
                latent = model.get_latent(filtered_data)
                latent = latent.data
                latent = latent[:,[pair[0], pair[1]]]
                plt.scatter(latent[:, 0], latent[:, 1], c=colors[label]["data"], label=str(label), alpha=0.75)

                delta = 0.025
                mean = np.mean(latent, axis=0)
                cov = np.cov(latent.T)
                x = np.arange(min(latent[:, 0]), max(latent[:, 0]), delta)
                y = np.arange(min(latent[:, 1]), max(latent[:, 1]), delta)
                X, Y = np.meshgrid(x, y)
                Z = multivariate_normal.pdf(np.array([zip(c,d) for c,d in zip(X,Y)]), mean=mean, cov=cov)
                plt.contour(X, Y, Z, colors=colors[label]["dist"])

            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

        # plot bounding box for the visualised manifold
        # boundaries are [[min_x, min_y],[max_x, max_y]]
        plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[0,1], boundaries[0,1]], 'r') # top line
        plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[1,1], boundaries[1,1]], 'r') # bottom line
        plt.plot([boundaries[0,0], boundaries[0,0]], [boundaries[0,1], boundaries[1,1]], 'r') # left line
        plt.plot([boundaries[1,0], boundaries[1,0]], [boundaries[0,1], boundaries[1,1]], 'r') # right line
        # major axes
        plt.plot([boundaries[0,0], boundaries[1,0]], [0,0], 'k')
        plt.plot([0,0], [boundaries[0,1], boundaries[1,1]], 'k')

        plt.grid()
        plt.xlabel("Z" + str(pair[0]))
        plt.ylabel("Z" + str(pair[1]))
        plt.savefig(filename + "_Z" + str(pair[0]) + "_Z" + str(pair[1]), bbox_inches="tight")
        plt.close()

    # # scatter datapoints and fit and overlay a distribution over each data label
    # counter = 0
    # for i, concept_group_labels in enumerate(labels)
    #     for label in sorted(set(concept_group_labels)):
    #         plt.figure(figsize=(10, 10))
    #         latent = latent_all[counter]
    #         plt.scatter(latent[:, 0], latent[:, 1], c=colors[label]["data"], label=str(label), alpha=0.75)

    #         if overlay:
    #             delta = 0.025
    #             mean = np.mean(latent, axis=0)
    #             cov = np.cov(latent.T)
    #             x = np.arange(min(latent[:, 0]), max(latent[:, 0]), delta)
    #             y = np.arange(min(latent[:, 1]), max(latent[:, 1]), delta)
    #             X, Y = np.meshgrid(x, y)
    #             Z = multivariate_normal.pdf(np.array([zip(c,d) for c,d in zip(X,Y)]), mean=mean, cov=cov)
    #             plt.contour(X, Y, Z, colors=colors[label]["dist"])
    #             plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

    #         # plot bounding box for the visualised manifold
    #         # boundaries are [[min_x, min_y],[max_x, max_y]]
    #         plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[0,1], boundaries[0,1]], 'r') # top line
    #         plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[1,1], boundaries[1,1]], 'r') # bottom line
    #         plt.plot([boundaries[0,0], boundaries[0,0]], [boundaries[0,1], boundaries[1,1]], 'r') # left line
    #         plt.plot([boundaries[1,0], boundaries[1,0]], [boundaries[0,1], boundaries[1,1]], 'r') # right line
    #         # major axes
    #         plt.plot([boundaries[0,0], boundaries[1,0]], [0,0], 'k')
    #         plt.plot([0,0], [boundaries[0,1], boundaries[1,1]], 'k')

    #         plt.grid()
    #         plt.savefig(filename + "_overlayed" + "_" + str(counter), bbox_inches="tight")
    #         plt.close()
    #         counter += 1


def plot_group_distribution(data=None, model=None, group_id=None, labels=None, colors=None, boundaries=None, 
                            overlay=False, n_groups=None, filename=None):
    
    labels = np.array([labels[i::n_groups] for i in range(n_groups)])
    concept_group_labels = labels[int(group_id)]

    plt.figure(figsize=(10, 10))
    for label in sorted(set(concept_group_labels)):
        indecies = [i for i, x in enumerate(concept_group_labels) if x == label]
        filtered_data = chainer.Variable(data.take(indecies, axis=0))
        latent = model.get_latent_mu(filtered_data)
        latent = latent.data[:,:2]

        plt.scatter(latent[:, 0], latent[:, 1], c=colors[label]["data"], label=str(label), alpha=0.75)

        if overlay:
            delta = 0.025
            mean = np.mean(latent, axis=0)
            cov = np.cov(latent.T)
            x = np.arange(min(latent[:, 0]), max(latent[:, 0]), delta)
            y = np.arange(min(latent[:, 1]), max(latent[:, 1]), delta)
            X, Y = np.meshgrid(x, y)
            Z = multivariate_normal.pdf(np.array([zip(c,d) for c,d in zip(X,Y)]), mean=mean, cov=cov)
            plt.contour(X, Y, Z, colors=colors[label]["dist"])
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

    # plot bounding box for the visualised manifold
    # boundaries are [[min_x, min_y],[max_x, max_y]]
    plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[0,1], boundaries[0,1]], 'r') # top line
    plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[1,1], boundaries[1,1]], 'r') # bottom line
    plt.plot([boundaries[0,0], boundaries[0,0]], [boundaries[0,1], boundaries[1,1]], 'r') # left line
    plt.plot([boundaries[1,0], boundaries[1,0]], [boundaries[0,1], boundaries[1,1]], 'r') # right line
    # major axes
    plt.plot([boundaries[0,0], boundaries[1,0]], [0,0], 'k')
    plt.plot([0,0], [boundaries[0,1], boundaries[1,1]], 'k')

    plt.grid()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# sample datapoints under the prior normal distribution and reconstruct
# samples_per_dimension has to be even
def plot_sampled_images(model=None, boundaries=None, samples_per_dimension=16, 
                        image_size=100, offset=10, image_channels=3, filename=None, figure_title=None):
        
        n_latent = model.n_latent

        # dimensions_pairs = list(itertools.combinations(range(n_latent), 2))
        dimensions_pairs = [(0,1)]
        for pair in dimensions_pairs:
            if 0 not in pair and 1 not in pair:
                continue
            rows = image_size * samples_per_dimension + offset * samples_per_dimension
            columns = image_size * samples_per_dimension + offset * samples_per_dimension
            figure = np.ones((rows, columns, image_channels))
            # major axes
            if image_channels == 1:
                line_pixel = [1]
            else:
                line_pixel = [0,0,1]
            quadrant_size = (samples_per_dimension / 2) * image_size + ((samples_per_dimension / 2) - 1) * offset
            figure[quadrant_size : quadrant_size + offset, :, :] = np.tile(line_pixel, (offset, (quadrant_size + offset) * 2, 1))
            figure[:, quadrant_size : quadrant_size + offset, :] = np.tile(line_pixel, ((quadrant_size + offset) * 2, offset, 1))
            
            # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
            # to produce values of the latent variables z, since the prior of the latent space is Gaussian
            # x and y are sptlit because of the way open cv has its axes
            columns = np.linspace(boundaries[1,1], boundaries[0,1], samples_per_dimension)
            rows = np.linspace(boundaries[0,0], boundaries[1,0], samples_per_dimension)

            for i, yi in enumerate(columns):
                for j, xi in enumerate(rows):
                    z_sample = np.zeros((1, n_latent))
                    z_sample[0, pair[0]] = xi
                    z_sample[0, pair[1]] = yi
                    # print(xi, yi, z_sample)
                    z_sample = np.array([[z_sample]]).astype(np.float32)
                    x_decoded = model.decode(chainer.Variable(z_sample)).data
                    image_sample = x_decoded.reshape(x_decoded.shape[1:])
                    image_sample = np.swapaxes(image_sample, 0, 2)
                    image_sample = image_sample.reshape(100, 100, 3)

                    figure[i * image_size + i * offset: (i + 1) * image_size + i * offset,
                           j * image_size + j * offset: (j + 1) * image_size + j * offset,
                           :] = image_sample

            figure = np.array(figure*255, dtype=np.uint8)

            plt.figure(figsize=(15,15))
            image = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            if figure_title:
                plt.title(figure_title + " Z0:[{0},{1}], Z1:[{2},{3}]".format(round(boundaries[0,0],1),
                                                                          round(boundaries[1,0],1),
                                                                          round(boundaries[0,1],1),
                                                                          round(boundaries[1,1],1)), 
                                                                          fontsize=20)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('Z' + str(pair[0]), fontsize=20)
            plt.ylabel('Z' + str(pair[1]), fontsize=20)
            plt.savefig(filename + '_Z' + str(pair[0]) + '_Z' + str(pair[1]), bbox_inches="tight")
            plt.close()

########################################
############# EVAL METRICS #############
########################################

def axes_alignment(data=None, labels=None, model=None, folder_name=None):

    labels = labels.tolist()
    labels = np.array([labels[i::2] for i in range(2)])
    print(labels)
    for index, group in enumerate(labels):
        for label in set(group):
            if label == "unknown":
                suffix = str(index)
            else:
                suffix = ""
            indecies = [i for i, x in enumerate(group) if x == label]
            filtered_data = chainer.Variable(data.take(indecies, axis=0))
            latent = model.get_latent_mu(filtered_data)
            latent = latent.data
            hinton_diagram(data=latent.T, label=label, folder_name=folder_name, suffix=suffix)
            # hinton_diagram(data=np.array([latent[:, i] for i in range(latent.shape[-1])]), label=label, folder_name=folder_name, suffix=suffix)
            # hinton_diagram(data=np.array([latent[:, i] for i in [0,1]]), label=label, folder_name=folder_name, suffix=suffix)

def hinton_diagram(data=None, label=None, folder_name=None, suffix=""):
        fig,ax = plt.subplots(1,1, figsize=(10,10))

        # print(label)
        # print(data)
        # print(data.shape)
        principal_axes = np.identity(data.shape[0])

        ax.patch.set_facecolor('lightgray')
        ax.set_aspect('equal', 'box')

        np.set_printoptions(precision=3)

        cov = np.cov(data)
        eig_val, eig_vec = np.linalg.eig(cov)
        eig_vec = eig_vec.T

        # print("Cov")
        # print(cov)
        # print("Eig vecs")
        # print(eig_vec)
        # print("Eig vals")
        # print(eig_val)
        # print("\n")

        ax.set_yticks([0,1,2,3,4,5,6,7],      minor=False)
        ax.set_yticklabels(['z' + str(i) for i, val in enumerate(eig_val)], minor=False, fontsize=40)

        # Customize minor tick labels
        ax.set_xticks([0,1,2,3,4,5,6,7],      minor=False)
        ax.set_xticklabels(['c' + str(i) + "\n" + str(round(val, 2)) for i, val in enumerate(eig_val)], minor=False, fontsize=40)
        # ax.set_xticklabels(['pc' + str(i) for i, val in enumerate(eig_val)], minor=False, fontsize=30)
        
        pairs = list(itertools.product(eig_vec, principal_axes))
        cosines = np.array([abs(cosine(x=p[0], y=p[1])) for p in pairs]).reshape(cov.shape)

        min_eig_value_i = eig_val.argmin()

        min_eig_value_max_cosine_i = cosines[min_eig_value_i].argmax()
        max_eig_value = eig_val.max()
        height, width = cosines.shape


        # print(min_eig_value_i)
        # print(cosines[min_eig_value_i])
        entropy_val = entropy([x / sum(cosines[min_eig_value_i,:]) for x in cosines[min_eig_value_i,:]], base=4)
        entropy_val += entropy([x / sum(cosines[:,min_eig_value_max_cosine_i]) for x in cosines[:,min_eig_value_max_cosine_i]], base=4)
        file = open(os.path.join(folder_name, "entropy"), "a")
        file.write("Entropy " + label + " " + str(entropy_val / 2.0) + "\n")
        file.close()
        
        fmt = '.1f'
        for (x, y), c in np.ndenumerate(cosines):
            # val = eig_val[x]
            if x == min_eig_value_i or y == min_eig_value_max_cosine_i:
                color = (1.0, 1.0, 1.0)
                text_color = (0.0, 0.0, 0.0)
            else:
                color = (0.0, 0.0, 0.0)
                text_color = (1.0, 1.0, 1.0)
            size = np.sqrt(c)
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)
            ax.text(x, y, format(c, fmt),
                     horizontalalignment="center",
                     color=text_color,
                     fontsize=10, fontweight="bold")

        ax.autoscale_view()
        ax.invert_yaxis()
        ax.set_title(label, fontweight="bold", fontsize=75)
        plt.savefig(os.path.join(folder_name, label + suffix + "_" + str(round(entropy_val / 2.0, 2)) + '_Hinton.png'), bbox_inches="tight")
        plt.close()

    
def cosine(x=None,y=None):
    return np.dot(x,y) / float(np.linalg.norm(x) * np.linalg.norm(y))


def test_time_classification(data_test=None, data_all=None, labels=None, unseen_labels=None, groups=None, 
                             boundaries=None, model=None, colors=None, folder_name=None):

    classifiers = {}
    stds = {}
    predicted_labels = []
    for key in sorted(groups.keys()):
        stds[key] = []
        classifiers[key] = []
        
    for key in sorted(groups.keys()):
        for label in groups[key]:
            indecies = [i for i, x in enumerate(labels) if x == label]
            filtered_data = chainer.Variable(data_test.take(indecies, axis=0))
            latent = model.get_latent_mu(filtered_data)
            latent = latent.data[:,:2]

            mean = np.mean(latent, axis=0)
            cov = np.cov(latent.T)
            classifiers[key].append({"label": label, "mean":mean[int(key)], "cov":cov[int(key),int(key)]})
        
        # sort the list by the value of the mean element
        classifiers[key] = sorted(classifiers[key], key=lambda k: k['mean']) 

    for key in sorted(classifiers.keys()):

        # intermediate unknown distributions
        classifier_tuples = zip(classifiers[key], classifiers[key][1:])

        # guarding unknown distributions
        lefmost = classifiers[key][0]
        rightmost = classifiers[key][-1]
        # boundaries are [[min_x, min_y],[max_x, max_y]]
        classifiers[key] += [{"label": "unknown", "mean": boundaries[0][int(key)], "cov": lefmost["cov"]}]
        classifiers[key] += [{"label": "unknown", "mean": boundaries[1][int(key)], "cov": rightmost["cov"]}]

        classifiers[key] += [{"label": "unknown", "mean": 0.5 * (cl1["mean"] + cl2["mean"]), "cov": 0.5 * (cl1["cov"] + \
                             cl2["cov"])} for (cl1, cl2) in classifier_tuples]


    # data_all = data_all[0::2]
    all_latent = model.get_latent_mu(data_all)

    # Show the 1D Gaussians per Group
    if len(unseen_labels) == 0:
        all_labels = labels
    else:
        all_labels = unseen_labels

    all_labels = np.array([all_labels[i::2] for i in [0,1]])

    for index,key in enumerate(sorted(classifiers.keys())):
        for label in list(set(all_labels[index])):
            range = np.arange(boundaries[0][int(key)], boundaries[1][int(key)], 0.001)
            if label in groups[key] or label[0] == "u":
                indecies = [i for i, x in enumerate(all_labels[index]) if x == label]
                filtered_data = chainer.Variable(np.repeat(data_all, len(groups.keys()), axis=0).take(indecies, axis=0))

                latent = model.get_latent_mu(filtered_data)

                plt.figure(figsize=(10,10))

                for cl in classifiers[key]:
                    color = colors["singular"][cl["label"]]["dist"]
                    plt.plot(range, norm.pdf(range, cl["mean"], cl["cov"]), color=color, label=cl["label"])
                    plt.plot([cl["mean"], cl["mean"]], [0, norm.pdf([cl["mean"]], cl["mean"], cl["cov"])], color='r', linestyle="--")
                    plt.xlim(boundaries[0][int(key)] - 2, boundaries[1][int(key)] + 2)
                x = latent[:, int(key)]
                y = np.zeros((1, len(latent)))
                plt.scatter(x.data, y, alpha=0.75, marker='o', label=label + "_data")
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
                plt.grid()
                plt.savefig(os.path.join(folder_name, label + str(index) + "_group_" + key + "_testime_distrobutions"), bbox_inches="tight")
                plt.close()

    for key in sorted(classifiers.keys()):
        points = all_latent[:, int(key)]
        stds[key] = [[{"label": c["label"], "value": abs(c["mean"] - point.data) / c["cov"]} for c in classifiers[key]] for point in points]
        stds[key] = map(lambda point_stds : sorted(point_stds, key=lambda k: k["value"])[0]["label"], stds[key])

    predicted_labels = np.array([stds[key] for key in sorted(stds.keys())])
    return predicted_labels


def label_analysis(labels=None, predictions=None, groups=None, model=None, folder_name=None, name=None):
    
    true_labels = []
    n_groups = len(groups)
    groups = copy.deepcopy(groups)
    for i in range(n_groups):
        groups[str(i)].append("unknown")
        true_labels.append(labels[i::n_groups])

    # at this point both true_labels and predictions are strings
    true_sets = [sorted(list(set(np.append(group, np.array(["unknown"]))))) for group in true_labels]
    pred_sets = [sorted(groups[group_key]) for group_key in sorted(groups.keys())]

    cms = []
    for i in range(len(predictions)):
        pred_per_group = predictions[i]
        true_per_group = true_labels[i]
        pred_set = pred_sets[i]
        true_set = true_sets[i]

        cm = np.zeros((len(true_set), len(pred_set)))

        for i in range(len(true_per_group)):
            label_t = true_per_group[i]
            label_p = pred_per_group[i]
            x = true_set.index(label_t)
            y = pred_set.index(label_p)
            cm[x,y] += 1

        cms.append(cm)

        file = open(os.path.join(folder_name, "f1"), "a")
        file.write(str(cm) + "\n")
        for label in true_set:

            if label not in true_set or label not in pred_set:
                continue

            x = true_set.index(label)
            y = pred_set.index(label)

            precision = cm[x,y] / float(sum(cm[x,:]) + 0.00001)
            recall = cm[x,y] / float(sum(cm[:,y]) + 0.00001)
            f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
            file.write("F1 " + label + " " + str(round(precision,2)) + " " + str(round(recall,2)) + " " + str(round(f1,2)) + "\n")
        file.close()

    plot_confusion_matrix(cms=cms, group_classes=zip(true_sets, pred_sets),
                          title="Confusion Matrix Singular" + " " + name,
                          folder_name=folder_name)


def plot_confusion_matrix(cms=None, group_classes=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          folder_name=None):

    print(group_classes)
    fig, subfigures = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
    for i, subfig in enumerate(subfigures):

        cm = cms[i]
        (true, pred) = group_classes[i]

        cm_norm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

        print(cm_norm)

        cax = subfig.imshow(cm_norm, interpolation='nearest', cmap=cmap)
        # subfig.set_title(i, fontweight="bold", fontsize=30)
        cbar = fig.colorbar(cax, ax=subfig)
        cbar.ax.tick_params(labelsize=25)   

        subfig.set_xticks(range(len(pred)), minor=False)
        subfig.set_xticklabels(pred, minor=False, fontsize=30)
        subfig.set_yticks(range(len(true)), minor=False)
        subfig.set_yticklabels(true, minor=False, fontsize=30)

        fmt = '.2f'
        thresh = cm.max() / 2.
        for x, y in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            size = 1.0
            rect = plt.Rectangle([y - size / 2, x - size / 2], size, size,
                                     facecolor=(0,0,0,0))
            subfig.add_patch(rect)
            subfig.text(y, x, format(cm_norm[x, y], fmt),
                     horizontalalignment="center",
                     color="white" if cm[x, y] > thresh else "black",
                     fontsize=25, fontweight='bold')

        subfig.autoscale_view()
        # subfig.set_ylabel('True label', fontsize=25)
        # subfig.set_xlabel('Predicted label', fontsize=25)

    fig.tight_layout()
    plt.savefig(os.path.join(folder_name, title + "_confusion_matrices" + '.png'), bbox_inches="tight")
    plt.close()
