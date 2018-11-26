#!/usr/bin/env python
"""
title           :object_segmentor.py
description     :Takes the processed kinect data and extracts the points from each pointcloud
                :corresponding to an individual object. The result is saved under 
                :rosbag_dumps/segmented_objects.npz
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :11/2018
python_version  :2.7.6
==============================================================================
"""

import struct
import copy
import time
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Object_Segmentor(object):

    def __init__(self, debug=False, args=None):
        self.debug = debug
        self.output = []
        self.counter = 0
        self.args = args
        self.bounds = {'x':[0.1, 1.3], 'y':[-0.5, 0.5], 'z':[0.0, 1.0]}

        # hue ranges - [0,180] - used when classifiying patches wrt color labels
        self.hues = [{},{}]
        self.hues[1]['blue'] = [90, 130]
        self.hues[0]['yellow'] = [10, 30]
        self.hues[0]['green'] = [30, 65]
        self.hues[0]['red'] = [170, 180]
        self.hues[0]['purple'] = [130, 170]
        self.hues[1]['orange'] = [0,10, 175, 180]


    def process_data(self):

        height = 540
        width = 960
        offset = 0
        margin_h = 50
        margin_w = 100
        
        image_bbox = [offset + margin_h, offset + height - margin_h, offset + margin_w, offset + width - margin_w]

        height -= 2*margin_h
        width -= 2*margin_w

        # half of the crop's size; we assume a square crop
        crop_size = 100

        # params for the bg subraction
        bg_patch_size = 20
        bg_threshold = 50

        for (xyz, bgr) in self.data:
            
            self.counter += 1
            print("{0} Clouds Segmented.".format(self.counter))

            xyz = xyz[image_bbox[0] : image_bbox[1], image_bbox[2] : image_bbox[3], :]
            bgr = bgr[image_bbox[0] : image_bbox[1], image_bbox[2] : image_bbox[3], :].astype(np.uint8)

            # filter only the carpet parts of the point cloud'
            image = bgr.copy()
            image_size = image.shape[:-1]
            number_of_tiles = (int(image_size[0] / bg_patch_size), int(image_size[1] / bg_patch_size), 1)
            
            backgrounds = []
            masks = []
            
            # remove the carpet
            # take patches for bg from the 4 courners
            top_left = image[0 : bg_patch_size, 0 : bg_patch_size, :]
            top_left[(top_left > 150).all(axis=2)] = 0
            backgrounds.append(np.tile(top_left, \
                                       number_of_tiles))

            bottom_left = image[image_size[0] - bg_patch_size : image_size[0], 0 : bg_patch_size, :]
            bottom_left[(bottom_left > 150).all(axis=2)] = 0
            backgrounds.append(np.tile(bottom_left, \
                                       number_of_tiles))

            top_right = image[0 : bg_patch_size, image_size[1] - bg_patch_size : image_size[1], :]
            top_right[(top_right > 150).all(axis=2)] = 0
            backgrounds.append(np.tile(top_right, \
                                       number_of_tiles))

            bottom_right = image[image_size[0] - bg_patch_size : image_size[0], image_size[1] - bg_patch_size : image_size[1], :]
            bottom_right[(bottom_right > 150).all(axis=2)] = 0
            backgrounds.append(np.tile(bottom_right, \
                                       number_of_tiles))

            # for each background sample generate a binary mask; after that combine them all
            for bg in backgrounds:
                mask = abs(image.astype(np.int32) - bg.astype(np.int32))
                mask =  (mask > bg_threshold).astype(np.int32)
                masks.append(mask)
            final_mask = np.ones((image_size[0], image_size[1], 3))
            for mask in masks:
                final_mask *= mask

            # for each pixel, if any of the channel have value = 1, the whole pixels is white in the mask
            final_mask = np.array(list(map(lambda row : list(map(lambda pixel : np.array([1,1,1]) if pixel[0] == 1 or pixel[1] == 1 or pixel[2] == 1 else pixel, row)), final_mask)), dtype=np.uint8)

            final_mask = final_mask[:,:,0]
            mask = self.fill_holes_get_max_cnt(final_mask)

            mask_extended = np.tile(mask[:,:,np.newaxis], (1,1,3))
            bgr = bgr * mask_extended

            # filter the white parts of the pointcloud
            mask = np.logical_and(np.any(bgr < 120, axis=2), np.any(bgr > 0, axis=2)).astype(np.uint8)
            # cv2.imshow("Pure Mask", mask * 255)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)

            # cv2.imshow("bg_sub_mask", mask_extended * 255)
            # cv2.imshow("Smoothed", mask * 255)
            # cv2.imshow("bgr", bgr)
            # cv2.imshow("bgr_masked", bgr * np.tile(mask[:,:,np.newaxis], (1,1,3)))
            # cv2.waitKey(0)

            mask_copy = copy.deepcopy(mask)
            contours, hier = cv2.findContours(mask_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(cnt) for cnt in contours]
            contours = [x for _,x in sorted(zip(areas,contours), reverse=True)]
            
            # extract the color and spatial information for each object
            new_mask = np.zeros((height, width))
            new_entry = {}
            for cnt_index, cnt in enumerate(contours[:self.args['no_object_groups']]):
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # makes sure that we are not trying to pass a negative or bigger than allowed index 
                # when slicing the xyz/bgr images
                bbox = [cY - crop_size, cY + crop_size, cX - crop_size, cX + crop_size]
                bbox = list(map(lambda x : 0 if x < 0 else x, bbox))

                maxes = np.repeat(image_bbox[1::2], 2)
                bbox = list(map(lambda (x, y) : y if x > y else x, zip(bbox, maxes)))

                xyz_crop = xyz[bbox[0] : bbox[1], bbox[2] : bbox[3], :]
                xyz_crop_norm = self.normalise_xyz(xyz_crop, bounds=self.bounds)
                bgr_crop = bgr[bbox[0] : bbox[1], bbox[2] : bbox[3], :]

                result = {}

                # filter out the white table artefacts (+ object shadows)
                xyz_crop_norm[(bgr_crop > 120).all(axis=2)] = 0
                bgr_crop[(bgr_crop > 120).all(axis=2)] = 0

                hsv_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)

                for hue, hue_values in self.hues[cnt_index].items():
                    mask = (np.logical_and(hsv_crop[...,0] > hue_values[0], hsv_crop[...,0] <= hue_values[1])).astype(np.uint8)
                    
                    # smooth out the mask
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


                    cv2.imshow("init mask", mask * 255)

                    if hue == "orange":
                        # revise the mask to include the low range of red-like colors
                        mask = (np.logical_or(np.logical_and(hsv_crop[...,0] > hue_values[0], hsv_crop[...,0] <= hue_values[1]), np.logical_and(hsv_crop[...,0] > hue_values[2], hsv_crop[...,0] <= hue_values[3]))).astype(np.uint8)
                        
                        kernel = np.ones((5, 5), np.uint8)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                        mask = self.fill_holes_get_max_cnt(mask)

                    # mask = self.fill_holes_get_max_cnt(mask)

                    # ignore any small patches
                    if np.sum(mask) < 500:
                        continue

                    xyz_object = xyz_crop_norm[mask == 1]
                    
                    # print(hue)
                    # print(np.sum(mask))

                    # hsv_copy = hsv_crop.copy()
                    # hsv_copy[mask == 0] = 0
                    # cv2.imshow("mask", mask.reshape(crop_size * 2, crop_size * 2) * 255)
                    # cv2.imshow("crop_masked", cv2.cvtColor(hsv_copy, cv2.COLOR_HSV2BGR))
                    # cv2.waitKey(0)

                    assert ((xyz_object <= 1).all() and (xyz_object >= 0).all()),\
                    "The data can not be normalised in the range [0,1] - Potentially bad bounds"

                    new_entry[hue] = xyz_object

            assert len(new_entry.items()) == self.args['no_objects'], \
            "Some objects were not extracted for cloud # {0}! {1}".format(self.counter, new_entry.keys())
                   
            self.output.append(new_entry)

    def fill_holes_get_max_cnt(self, mask, fill=-1):
        """Given a binary mask, fills in any holes in the contours, selects the contour with max area
        and returns a mask with only it drawn + its bounding box
        """
        canvas = np.zeros(mask.shape, dtype=np.uint8)
        cnts, hier = cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cnt) for cnt in cnts]

        # for (cnt, area) in sorted(zip(cnts, areas), key=lambda x: x[1], reverse=True):
        #     cv2.drawContours(canvas,[cnt],0,1,-1)

        (cnt, area) = sorted(zip(cnts, areas), key=lambda x: x[1], reverse=True)[0]
        cv2.drawContours(canvas,[cnt],0,1,fill)

        return canvas.astype(np.uint8)


    def normalise_xyz(self, xyz_points, bounds={}):
        
        xyz_points_norm = np.zeros(xyz_points.shape)
        xyz_points_norm[...,0] = (xyz_points[...,0] - bounds['x'][0]) / (bounds['x'][1] - bounds['x'][0])
        xyz_points_norm[...,1] = (xyz_points[...,1] - bounds['y'][0]) / (bounds['y'][1] - bounds['y'][0])
        xyz_points_norm[...,2] = (xyz_points[...,2] - bounds['z'][0]) / (bounds['z'][1] - bounds['z'][0])

        mask = np.logical_and((xyz_points_norm <= 1).all(axis=2), (xyz_points_norm >= 0).all(axis=2)).astype(np.uint8)
        mask = np.tile(mask.reshape(xyz_points_norm.shape[0], xyz_points_norm.shape[1], 1), (1, 1, 3))
        xyz_points_norm = xyz_points_norm * mask

        return xyz_points_norm


    def plot_xyz(self, xyz_points):

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


    def load_processed_rosbag(self, path):
        self.data = np.load(path)['arr_0']

    def save_to_npz(self, path):
        np.savez(path, self.output)


if __name__ == '__main__':

    args = {'no_objects':6, 'no_object_groups':2}
    segmentor = Object_Segmentor(debug=True, args=args)
    segmentor.load_processed_rosbag("/home/yordan/pr2_ws/src/spatial_relations_experiments/kinect_processor/rosbag_dumps/processed_rosbag.npz")
    segmentor.process_data()

    print("Final count: "+ str(segmentor.counter))
    segmentor.output = np.array(segmentor.output)
    segmentor.save_to_npz("/home/yordan/pr2_ws/src/spatial_relations_experiments/kinect_processor/rosbag_dumps/segmented_objects.npz")
    print("NPZ saved")