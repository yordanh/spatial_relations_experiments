#!/usr/bin/env python
"""
title           :data_generator.py
description     :Loads the spatial dataset contained in numpy arrays under train,unseen,ulabelled 
                :folders under learning_experiments/data/.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :10/2018
python_version  :2.7.6
==============================================================================
"""

import os
import cv2
import numpy as np

# remove the following imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from config_parser import ConfigParser

class DataGenerator(object):
    def __init__(self, label_mode=None):
        self.label_mode = label_mode

    def generate_dataset(self, ignore=[], args=None):
        
        # folder_name_train = "data/" + args.data + "/train/"
        # folder_name_unseen = "data/" + args.data + "/unseen/"
        # folder_name_unlabelled = "data/" + args.data + "/unlabelled/"
        folder_name_train = "data/train/"
        folder_name_unseen = "data/unseen/"
        folder_name_unlabelled = "data/unlabelled/"
        data_split = 0.8
        crop_size = 200
        data_dimensions = [crop_size, crop_size, 3]

        seed = 0
        np.random.seed(seed)

        # config_parser = ConfigParser(os.path.join(args.config, "config.json"))
        # config_parser = ConfigParser(os.path.join("config", "config.json"))
        # labels = config_parser.parse_labels()
        # groups = config_parser.parse_groups()

        groups = {i : group.replace('.npz','').split('_')[:-1] for i,group in enumerate(sorted(os.listdir(folder_name_train)))}

        train_b0 = []
        train_b1 = []
        train_labels = []
        train_vectors = []
        train_masks = []
        
        test_b0 = []
        test_b1 = []
        test_labels = []
        test_vectors = []
        test_masks = []

        unseen_b0 = []
        unseen_b1 = []
        unseen_labels = []

        array_list_train = os.listdir(folder_name_train)

        train_vectors = [[] for group in groups]
        train_masks = [[] for group in groups]

        test_vectors = [[] for group in groups]
        test_masks = [[] for group in groups]

        if "train" not in ignore:
            for array_name in array_list_train:
                pair_list = np.load(os.path.join(folder_name_train, array_name))
                labels = list(pair_list['label'])
                seed += 1
                np.random.seed(seed)
                number_of_pairs = len(pair_list['branch_0'])
                train_n = int(data_split * number_of_pairs)
                test_n = number_of_pairs - train_n
                train_indecies = np.random.choice(range(number_of_pairs), train_n, replace=False)
                test_indecies = filter(lambda x : x not in train_indecies, range(number_of_pairs))

                print("Processing TRAINING array {0}/{1} with {2} pairs".format(array_list_train.index(array_name) + 1, 
                                                                                  len(array_list_train), 
                                                                                  number_of_pairs))

                # remove _X.npz suffix from the name
                array_name = array_name.split('_')[:-1]
                array_name = '_'.join(array_name)

                train_b0 += list(np.take(pair_list['branch_0'], train_indecies, axis=0))
                train_b1 += list(np.take(pair_list['branch_1'], train_indecies, axis=0))
                vectors = list(np.take(pair_list['label'], train_indecies, axis=0))                      

                train_labels += [array_name.split('_')[x] for x in vectors]

                for i in groups:
                    label = filter(lambda x : x in groups[i], array_name.split('_'))
                    if label != []:
                        train_vectors[i] += vectors
                        train_masks[i] += [1] * train_n
                    else:
                        label = 0
                        train_vectors[i] += list(np.tile(label, (train_n)))
                        train_masks[i] += [0] * train_n



                test_b0 += list(np.take(pair_list['branch_0'], test_indecies, axis=0))
                test_b1 += list(np.take(pair_list['branch_1'], test_indecies, axis=0))
                vectors = list(np.take(pair_list['label'], test_indecies, axis=0))

                test_labels += [array_name.split('_')[x] for x in vectors]

                for i in groups:
                    label = filter(lambda x : x in groups[i], array_name.split('_'))
                    if label != []:
                        test_vectors[i] += vectors
                        test_masks[i] += [1] * test_n
                    else:
                        label = 0
                        test_vectors[i] += list(np.tile(label, (test_n)))
                        test_masks[i] += [0] * test_n









        # # unlabelled datapoints
        # if os.path.exists(folder_name_unlabelled) and "unlabelled" not in ignore:
        #     array_list_unlabelled = os.listdir(folder_name_unlabelled)
        #     for array_name in array_list_unlabelled:
        #         # pair_list = np.load(os.path.join("/home/yordan/pr2_ws/src/spatial_relations_experiments/learning_experiments/data/unlabelled/", array_name))
        #         pair_list = np.load(os.path.join(folder_name_unlabelled, array_name))
        #         seed += 1
        #         np.random.seed(seed)
        #         number_of_pairs = len(pair_list)
        #         train_n = int(data_split * number_of_pairs)
        #         test_n = number_of_pairs - train_n
        #         train_indecies = np.random.choice(range(number_of_pairs), train_n, replace=False)
        #         test_indecies = filter(lambda x : x not in train_indecies, range(number_of_pairs))

        #         print("Processing UNLABELLED array_name {0}/{1} with {2} pairs".format(array_list_unlabelled.index(array_name) + 1, 
        #                                                                             len(array_list_unlabelled), 
        #                                                                             number_of_pairs))

        #         train_b0 += list(np.take(pair_list['branch_0'], train_indecies, axis=0))
        #         train_b1 += list(np.take(pair_list['branch_1'], train_indecies, axis=0))

        #         train_labels += array_name.split('_')[:3] * train_n
        #         train_masks += [0] * train_n

        #         for i, group in enumerate(groups):
        #             label = 0
        #             train_vectors[i] += list(np.tile(label, (train_n)))

        #         test_b0 += list(np.take(pair_list['branch_0'], test_indecies, axis=0))
        #         test_b1 += list(np.take(pair_list['branch_1'], test_indecies, axis=0))

        #         test_labels += array_name.split('_')[:3] * test_n
        #         test_masks += [0] * test_n

        #         for i, group in enumerate(groups):
        #             label = 0
        #             test_vectors[i] += list(np.tile(label, (test_n)))




        # unseen datapoints
        if os.path.exists(folder_name_unseen) and "unseen" not in ignore:
            array_list_unseen = os.listdir(folder_name_unseen)
            for array_name in array_list_unseen:
                pair_list = np.load(os.path.join(folder_name_unseen, array_name))
                
                seed += 1
                np.random.seed(seed)
                number_of_pairs = len(pair_list['branch_0'])
                unseen_n = number_of_pairs                                     

                print("Processing UNSEEN array_name {0}/{1} with {2} pairs".format(array_list_unseen.index(array_name) + 1, 
                                                                                len(array_list_unseen), 
                                                                                number_of_pairs))

                # remove _X.npz suffix from the name
                array_name = array_name.split('_')[:-1]
                array_name = '_'.join(array_name)

                unseen_b0 += list(np.take(pair_list['branch_0'], range(unseen_n), axis=0))
                unseen_b1 += list(np.take(pair_list['branch_1'], range(unseen_n), axis=0))
                vectors = list(np.take(pair_list['label'], range(unseen_n), axis=0)) 
                
                unseen_labels += [array_name.split('_')[x] for x in vectors]



        # print("Train Vectors Shape: {}".format(np.array(train_vectors).shape))
        # print("Train Vectors: {}".format(np.array(train_vectors)))
        # print("Train Labels Shape: {}".format(np.array(train_labels).shape))
        # print("Train Labels: {}".format(np.array(train_labels)))
        # print("Train Masks Shape: {}".format(np.array(train_masks).shape))
        # print("Train Masks: {}".format(np.array(train_masks)))

        # print("Test Vectors: {}".format(np.array(test_vectors)))
        # print("Test Labels: {}".format(np.array(test_labels)))
        # print("Test Masks: {}".format(np.array(test_masks)))

        # print("Unseen Vectors: {}".format(np.array(unseen_vectors)))
        # print("Unseen Labels: {}".format(np.array(unseen_labels)))

        # train = np.array(train, dtype=np.float32) / 255.
        
        # lens = [len(train_b0[i]) for i in range(len(train_b0))]



        for i in range(len(train_b0)):
            if len(train_b0[i]) != 200:
                print(i, len(train_b0[i]))
                
        train_b0 = np.array(train_b0, dtype=np.float32)
        train_b1 = np.array(train_b1, dtype=np.float32)
        train_labels = np.array(train_labels)
        train_vectors = np.array(train_vectors)
        train_masks = np.array(train_masks)

        test_b0 = np.array(test_b0, dtype=np.float32)
        test_b1 = np.array(test_b1, dtype=np.float32)
        test_labels = np.array(test_labels)
        test_vectors = np.array(test_vectors)
        test_masks = np.array(test_masks)
        
        unseen_b0 = np.array(unseen_b0, dtype=np.float32)
        unseen_b1 = np.array(unseen_b1, dtype=np.float32)
        unseen_labels = np.array(unseen_labels)

        train_b0 = train_b0.reshape([len(train_b0)] + data_dimensions)
        test_b0 = test_b0.reshape([len(test_b0)] + data_dimensions)
        unseen_b0 = unseen_b0.reshape([len(unseen_b0)] + data_dimensions)
        train_b0 = np.swapaxes(train_b0, 1 ,3)
        test_b0 = np.swapaxes(test_b0, 1 ,3)
        unseen_b0 = np.swapaxes(unseen_b0, 1 ,3)

        train_b1 = train_b1.reshape([len(train_b1)] + data_dimensions)
        test_b1 = test_b1.reshape([len(test_b1)] + data_dimensions)
        unseen_b1 = unseen_b1.reshape([len(unseen_b1)] + data_dimensions)
        train_b1 = np.swapaxes(train_b1, 1 ,3)
        test_b1 = np.swapaxes(test_b1, 1 ,3)
        unseen_b1 = np.swapaxes(unseen_b1, 1 ,3)

        #augment the training, testing, unseen datapoints with their labels
        train_masks_and_vectors = np.append(train_masks, train_vectors, axis=0)
        test_masks_and_vectors = np.append(test_masks, test_vectors, axis=0)
        train_concat = zip(train_b0, train_b1, *train_masks_and_vectors)
        test_concat = zip(test_b0, test_b1, *test_masks_and_vectors)

        result = []
        result.append(train_b0)
        result.append(train_b1)
        result.append(train_labels)
        result.append(train_concat)
        result.append(train_vectors)

        result.append(test_b0)
        result.append(test_b1)
        result.append(test_labels)
        result.append(test_concat)
        result.append(test_vectors)

        result.append(unseen_b0)
        result.append(unseen_b1)
        result.append(unseen_labels)

        result.append(groups)

        return result

def plot_xyz(branch_0, branch_1, labels, vectors):

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

    plt.show()


if __name__ == "__main__":
    data_generator = DataGenerator()
    result = data_generator.generate_dataset()