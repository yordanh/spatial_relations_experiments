#!/usr/bin/env python
"""
title           :kinect_to_train.py
description     :Takes the processed kinect data in rosbad_dumps/processed_rosbag.npz
				:segments the objects, annotates pairs and saves to training data 
				:in learning_experiments/data/train/<rel_name>.npz
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :11/2018
python_version  :2.7.6
==============================================================================
"""

from object_segmentor import Object_Segmentor
from spatial_annotator import Spatial_Annotator

if __name__ == "__main__":

	PROCESSED_ROSBAG = "rosbag_dumps/processed_rosbag.npz"
	SEGMENTED_CLOUDS = "rosbag_dumps/segmented_objects.npz"
	ANNOTATOR_CONFIG = "config/config.json"
	TRAIN_DATA = "../learning_experiments/data/train/"
	args = {'no_objects':6, 'no_object_groups':2}

	segmentor = Object_Segmentor(verbose=False, args=args)
	print(SEGMENTED_CLOUDS + " LOADING\n")
	segmentor.load_processed_rosbag(PROCESSED_ROSBAG)
	segmentor.process_data()
	segmentor.save_to_npz(SEGMENTED_CLOUDS)
	print(SEGMENTED_CLOUDS + " SAVED\n")

	annotator = Spatial_Annotator(ANNOTATOR_CONFIG)
	print(TRAIN_DATA + " LOADING\n")
	annotator.load_segmented_clouds(SEGMENTED_CLOUDS)
	annotator.annotate()
	annotator.save_to_npz(path=TRAIN_DATA)
	print(TRAIN_DATA + " saved\n")