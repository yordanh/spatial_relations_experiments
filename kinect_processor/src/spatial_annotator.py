#!/usr/bin/env python
"""
title           :spatial_annotator.py
description     :Given user instructions, saves crops from the sensory feed in a numpy array under 
                :learning_experiments/data/.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :11/2018
python_version  :2.7.6
==============================================================================
"""

import copy
import time
import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Data_Annotator(object):

    def __init__(self, config_name):

        # uses the information from the instructions to construct the conceptual groups
        self.data = {}
        self.rosbad_dump = None
            
        config_file = open(config_name, "r")
        config = json.load(config_file)
        for instruction in config['instructions']:
            # instruction = parameters['instruction ' + str(i)]
            labels = instruction.split()[1].split('/')

            array_key = '_'.join(labels) + '_' + '_'.join(instruction.split()[:1] + instruction.split()[2:])
            # contains two arrays for the inputs of the two network branches
            self.data[array_key] = [[],[]]

        print(self.data)


    # given the linguistic instruction, decide which crop goes to which branch of the net and what
    # spatial label is the tuple given
    def annotate(self, data, debug=False):
        
        no_input_pairs = len(data)

        for entry in self.rosbad_dump:
            xyz_crops = entry[0]
            bgr_crops = entry[1]
            bgr_out = [[],[]]

            for array_key in self.data:

                # we expect the last two symbols in the array name to be colors
                color_labels = array_key.split('_')[-2:]
                
                # print(color_labels)
                # print(bgr_crops.shape)
                # cv2.imshow("0", bgr_crops[0].astype(np.uint8))
                # cv2.imshow("1", bgr_crops[1].astype(np.uint8))
                # cv2.imshow("2", bgr_crops[2].astype(np.uint8))
                # cv2.imshow("3", bgr_crops[3].astype(np.uint8))
                # cv2.waitKey(0)

                for bgr_index, bgr_crop in enumerate(bgr_crops):

                    for index, color_label in enumerate(color_labels):
                        if self.is_color(bgr_crop.astype(np.uint8), color_label):
                            self.data[array_key][index].append(xyz_crops[bgr_index])
                            bgr_out[index] = bgr_crop.astype(np.uint8)

                            break

                if debug:
                    print(array_key)
                    cv2.imshow("first", bgr_out[0])
                    cv2.imshow("second", bgr_out[1])
                    print('\n')
                    cv2.waitKey(1000)

            # keys = self.data.keys()
            # a = self.data[keys[0]][0][0]
            # b = self.data[keys[1]][0][0]
            # print(a == b)
            # exit()

        no_output_pairs = 0
        for key in self.data.keys():
            no_output_pairs += len(self.data[key][0])
            no_output_pairs += len(self.data[key][1])

        no_output_pairs = no_output_pairs / float(len(self.data.keys()))

        assert (no_input_pairs * 2 == no_output_pairs),\
        "Input pairs - {0} | Output pairs - {1}; Potential bad color ranges".format(no_input_pairs * 2, no_output_pairs)


    def load_rosbag_dump(self, rosbad_dump_dir):
        self.rosbad_dump = np.load(rosbad_dump_dir)['arr_0']


    def save_to_npz(self):

        for key in self.data.keys():
            output = {"branch_0":self.data[key][0], "branch_1":self.data[key][1]}
            np.savez(os.path.join("/home/yordan/pr2_ws/src/spatial_relations_experiments/learning_experiments/data/train/", key + ".npz"), **output)


if __name__ == "__main__":

    ROSBAG_DUMP = "../rosbad_dump/"
    CONFIG = "../config/config.txt"

    annotator = Data_Annotator()

    annotator.load_rosbag_dump()

    annotator.annotate()

    annotator.save_to_npz()