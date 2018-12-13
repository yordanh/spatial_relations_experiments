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

import argparse
import copy
import time
import numpy as np
import cv2
import os
import os.path as osp
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Spatial_Annotator(object):

    def __init__(self, args=None):

        # uses the information from the instructions to construct the conceptual groups
        self.data = {}
        self.rosbad_dump = None

        self.dtype = args.dtype
        self.scene = args.scene
        self.scenes_path = args.scenes_path
            
        config_file = open(osp.join(self.scenes_path, self.scene, 'config.json'), "r")
        config = json.load(config_file)

        self.concept_groups = config['groups']
        self.instructions = config['instructions']
        self.data = {'_'.join(spatial_labels) : [[],[],[]] for spatial_labels in config['groups'].values()}
        self.data['unseen'] = [[],[],[]]

        print(self.data)
        print(self.instructions)


    # given the linguistic instruction, decide which crop goes to which branch of the net and what
    # spatial label is the tuple given
    def annotate(self, debug=False):
        
        if self.dtype == 'train':
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
        
        # assumes the unseen data is always clouds with two objects in the scene
        # TODO - generalise that to multiple objects and resultant possble pairs
        elif self.dtype == 'unseen':
            for entry in self.rosbad_dump:
                keys = entry.keys()
                self.data['unseen'][0].append(entry[keys[0]])
                self.data['unseen'][1].append(entry[keys[1]])
                self.data['unseen'][2].append(0)


        # remove any instructions that were not grounded in the observations
        self.data = {key : self.data[key] for key in self.data if self.data[key] != [[],[],[]]}


    def load_segmented_clouds(self, scenes_path='scenes'):
        self.rosbad_dump = np.load(osp.join(scenes_path, self.scene, 'segmented_objects.npz'))['arr_0']


    def save_to_npz(self, output_path='../learning_experiments/data'):

        output_path = osp.join(output_path, self.dtype)
        for key in self.data.keys():
            output = {"branch_0":self.data[key][0], "branch_1":self.data[key][1], "label":self.data[key][2]}
            
            npz_list = [x for x in os.listdir(output_path) if key == '_'.join(x.split('_')[:-1])]
            if npz_list == []:
                index = 0
            else:
                index = sorted([int(x.replace('.npz', '').split('_')[-1]) for x in npz_list])[-1] + 1

            np.savez(os.path.join(output_path, key + '_' + str(index) + ".npz"), **output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Annotate the data for a particular scene')
    parser.add_argument('--dtype', default='train', type=str,
                        help='Type of data to annotate')
    parser.add_argument('--scenes_path', default='scenes', type=str,
                        help='Folder in which the data for all scenes is kept')
    parser.add_argument('--scene', '-sc', default='0',
                        help='Index for a scene/setup')
    args = parser.parse_args()

    annotator = Spatial_Annotator(args=args)

    annotator.load_segmented_clouds()
    annotator.annotate()
    annotator.save_to_npz()