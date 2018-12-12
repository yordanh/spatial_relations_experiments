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

class Spatial_Annotator(object):

    def __init__(self, config_name):

        # uses the information from the instructions to construct the conceptual groups
        self.data = {}
        self.rosbad_dump = None
            
        config_file = open(config_name, "r")
        config = json.load(config_file)

        self.concept_groups = config['groups']
        self.instructions = config['instructions'].values() 
        self.data = {'_'.join(spatial_labels) : [[],[],[]] for spatial_labels in config['groups'].values()}
        self.data['unseen'] = [[],[],[]]

        print(self.data)
        print(self.instructions)


    # given the linguistic instruction, decide which crop goes to which branch of the net and what
    # spatial label is the tuple given
    def annotate(self, debug=False):
        
        # no_input_clouds = len(self.rosbad_dump)

        for i, instruction in enumerate(self.instructions):

            print("{0}/{1} instructions processed".format(i, len(self.instructions)))

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

        # remove any instructions that were not grounded in the observations
        self.data = {key : self.data[key] for key in self.data if self.data[key] != [[],[],[]]}


    # assumes the unseen data is always clouds with two objects in the scene
    # TODO - generalise that to multiple objects and resultant possble pairs
    def annotate_unseen(self, debug=False):
        for entry in self.rosbad_dump:
            keys = entry.keys()
            self.data['unseen'][0].append(entry[keys[0]])
            self.data['unseen'][1].append(entry[keys[1]])
            self.data['unseen'][2].append(0)

        # remove any instructions that were not grounded in the observations
        self.data = {key : self.data[key] for key in self.data if self.data[key] != [[],[],[]]}


    def load_segmented_clouds(self, rosbad_dump_dir):
        self.rosbad_dump = np.load(rosbad_dump_dir)['arr_0']


    def save_to_npz(self, path):

        for key in self.data.keys():
            output = {"branch_0":self.data[key][0], "branch_1":self.data[key][1], "label":self.data[key][2]}
            np.savez(os.path.join(path, key + ".npz"), **output)


if __name__ == "__main__":

    PATH = 'scenes'

    annotator = Spatial_Annotator(PATH)

    annotator.load_segmented_clouds(SEGMENTED_CLOUDS)

    # annotator.annotate()
    if args['mode'] == 'train':
        annotator.annotate()
        annotator.save_to_npz(path="/data/learning_experiments/data/train/")
    else:
        annotator.annotate_unseen()
        annotator.save_to_npz(path="/data/learning_experiments/data/unseen/")

    # annotator.save_to_npz(path="/home/yordan/pr2_ws/src/spatial_relations_experiments/learning_experiments/data/unseen/")
    # annotator.save_to_npz(path="/data/learning_experiments/data/unseen/")