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

        self.concept_groups = config['groups']
        self.instructions = config['instructions'].values() 
        self.data = {'_'.join(spatial_labels) : [[],[],[]] for spatial_labels in config['groups'].values()}

        print(self.data)
        print(self.instructions)


    # given the linguistic instruction, decide which crop goes to which branch of the net and what
    # spatial label is the tuple given
    def annotate(self, debug=False):
        
        # no_input_clouds = len(self.rosbad_dump)

        for instruction in self.instructions:

            instruction = instruction.split()
            branch_0_label = instruction[0]
            branch_1_label = instruction[2]
            spatial_label = instruction[1]

            label = None
            concept_label = None
            for concept_group in self.concept_groups.values():
                if spatial_label in concept_group:
                    label = concept_group.index(spatial_label)
                    concept_label = '_'.join(concept_group)
                    break

            for entry in self.rosbad_dump:

                if not (branch_0_label in entry and branch_1_label in entry):
                    continue

                self.data[concept_label][0].append(entry[branch_0_label])
                self.data[concept_label][1].append(entry[branch_1_label])
                self.data[concept_label][2].append(label)
                    
                # if debug:
                #     print(array_key)
                #     cv2.imshow("first", bgr_out[0])
                #     cv2.imshow("second", bgr_out[1])
                #     print('\n')
                #     cv2.waitKey(1000)

        # remove any instructions that were not grounded in the observations
        self.data = {key : self.data[key] for key in self.data if self.data[key] != [[],[],[]]}

        # no_output_clouds = 0
        # for key in self.data.keys():
        #     no_output_clouds += len(self.data[key][0]) * 0.5
        #     no_output_clouds += len(self.data[key][1]) * 0.5

        # no_output_clouds = no_output_clouds / float(len(self.data.keys()))

        # assert (no_input_clouds == no_output_clouds),\
        # "Input clouds - {0} | Output clouds - {1};".format(no_input_clouds, no_output_clouds)


    def load_rosbag_dump(self, rosbad_dump_dir):
        self.rosbad_dump = np.load(rosbad_dump_dir)['arr_0']


    def save_to_npz(self):

        for key in self.data.keys():
            output = {"branch_0":self.data[key][0], "branch_1":self.data[key][1], "label":self.data[key][2]}
            np.savez(os.path.join("/home/yordan/pr2_ws/src/spatial_relations_experiments/learning_experiments/data/train/", key + ".npz"), **output)


if __name__ == "__main__":

    ROSBAG_DUMP = "rosbag_dump/rosbag_dump.npz"
    CONFIG = "config/config.json"

    annotator = Data_Annotator(CONFIG)

    annotator.load_rosbag_dump(ROSBAG_DUMP)

    annotator.annotate()

    annotator.save_to_npz()