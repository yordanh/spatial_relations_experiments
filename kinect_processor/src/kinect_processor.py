#!/usr/bin/env python
"""
title           :kinect_processor.py
description     :Processes raw sensory data captured from a Kinect 2 under the ROS framework. 
                :saves crops from the sensory feed in a numpy array under ../rosbag_dump.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :10/2018
python_version  :2.7.6
==============================================================================
"""

# ROS dependencies
import rospy
import tf
from tf import transformations
from sensor_msgs.msg import PointCloud2, PointField

# Misc Dependencies
import struct
import copy
import time
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


############################
###### Helper Methods ######
############################

fmt_full = ''

_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

tf_listener = tf.TransformListener()

def pointcloud2_to_array_dummy(msg):
    global fmt_full
    if not fmt_full:
        fmt = _get_struct_fmt(msg)
        fmt_full = '>' if msg.is_bigendian else '<' + fmt.strip('<>')*msg.width*msg.height
    unpacker = struct.Struct(fmt_full)
    unpacked = np.asarray(unpacker.unpack_from(msg.data))

    return unpacked.reshape(msg.height, msg.width, len(msg.fields))

def _get_struct_fmt(cloud, field_names=None):
    fmt = '>' if cloud.is_bigendian else '<'
    offset = 0
    for field in (f for f in sorted(cloud.fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print >> sys.stderr, 'Skipping unknown PointField datatype [%d]' % field.datatype
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


######################
### From numpy_pc2 ###
######################

# Functions for working with PointCloud2.
__docformat__ = "restructuredtext en"


# prefix to the names of dummy fields we add to get byte alignment correct. this needs to not
# clash with any actual field names
DUMMY_FIELD_PREFIX = '__'
 
# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')),
                 (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')),
                 (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')),
                 (PointField.FLOAT64, np.dtype('float64'))]
 
pftype_to_nptype = dict(type_mappings)
 
# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}


def pointcloud2_to_dtype(cloud_msg):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []

    # print(cloud_msg.fields)
    # print(cloud_msg.point_step)

    for f in cloud_msg.fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1
        np_dtype_list.append((f.name, pftype_to_nptype[f.datatype]))
        offset += pftype_sizes[f.datatype]
 
    # might be extra padding between points
    while offset < cloud_msg.point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1
 
    return np_dtype_list


def pointcloud2_to_array(cloud_msg, split_rgb=False, remove_padding=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray
 
    Reshapes the returned array to have shape (height, width), even if the height is 1.
 
    The reason for using np.fromstring rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = pointcloud2_to_dtype(cloud_msg)
 
    # print(dtype_list)

    # parse the cloud into an array
    cloud_arr = np.fromstring(cloud_msg.data, dtype_list)

    # print(type(cloud_arr[0]))
    # print(np.array(cloud_arr[0]))
 
    # remove the dummy fields that were added
    if remove_padding:
        cloud_arr = cloud_arr[
            [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

    if split_rgb:
        cloud_arr = split_rgb_field(cloud_arr)
 
    return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


def split_rgb_field(cloud_arr):
    '''Takes an array with a named 'rgb' float32 field, and returns an array in which
    this has been split into 3 uint 8 fields: 'r', 'g', and 'b'.
 
    (pcl stores rgb in packed 32 bit floats)
    '''
    rgb_arr = cloud_arr['rgb'].copy()
    rgb_arr.dtype = np.uint32
    r = np.asarray((rgb_arr >> 16) & 255, dtype=np.uint8)
    g = np.asarray((rgb_arr >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb_arr & 255, dtype=np.uint8)
 
    # create a new array, without rgb, but with r, g, and b fields
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if not field_name == 'rgb':
            new_dtype.append((field_name, field_type))
    new_dtype.append(('r', np.uint8))
    new_dtype.append(('g', np.uint8))
    new_dtype.append(('b', np.uint8))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)
 
    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        # print(field_name)
        if field_name == 'r':
            new_cloud_arr[field_name] = r
        elif field_name == 'g':
            new_cloud_arr[field_name] = g
        elif field_name == 'b':
            new_cloud_arr[field_name] = b
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]
    return new_cloud_arr

def get_xyzbgr_points(cloud_array, remove_nans=True):
    '''Pulls out x, y, z and r, g, b columns from the cloud recordarray, and returns
    a 2 3xN matrices (one with floats and one with unsigned 8-bit integets.
    '''

    # remove crap points
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z']) & \
               np.isfinite(cloud_array['r']) & np.isfinite(cloud_array['g']) & np.isfinite(cloud_array['b'])
        cloud_array = cloud_array[mask]
 
    # pull out x, y, and z values
    xyz_points = np.zeros(list(cloud_array.shape) + [3], dtype=np.float32)
    xyz_points[...,0] = cloud_array['x']
    xyz_points[...,1] = cloud_array['y']
    xyz_points[...,2] = cloud_array['z']

    # pull out r, g, and b values
    bgr_points = np.zeros(list(cloud_array.shape) + [3], dtype=np.uint8)
    bgr_points[...,0] = cloud_array['b']
    bgr_points[...,1] = cloud_array['g']
    bgr_points[...,2] = cloud_array['r']
 
    return xyz_points, bgr_points


def transform_xyz(xyz_points, target_frame="base_link", height=None, width=None):

    tf_listener.waitForTransform(target_frame, "/head_mount_kinect2_ir_optical_frame", rospy.Time(), rospy.Duration(4.0))
    translation, rotation = tf_listener.lookupTransform(target_frame, "/head_mount_kinect2_ir_optical_frame", rospy.Duration(0))
    mat44 = np.dot(transformations.translation_matrix(translation), transformations.quaternion_matrix(rotation))
    xyz_homogeneous_points = np.hstack((xyz_points.reshape(height*width, 3), np.ones((height*width, 1)))).reshape(height, width, 4)
    xyz_points_transformed = np.tensordot(xyz_homogeneous_points, mat44, axes=((2),(1)))[:,:,:3]

    return xyz_points_transformed


def normalise_xyz(xyz_points, bounds={}):
    
    xyz_points_norm = np.zeros(xyz_points.shape)
    xyz_points_norm[...,0] = (xyz_points[...,0] - bounds['x'][0]) / (bounds['x'][1] - bounds['x'][0])
    xyz_points_norm[...,1] = (xyz_points[...,1] - bounds['y'][0]) / (bounds['y'][1] - bounds['y'][0])
    xyz_points_norm[...,2] = (xyz_points[...,2] - bounds['z'][0]) / (bounds['z'][1] - bounds['z'][0])

    mask = np.logical_and((xyz_points_norm <= 1).all(axis=2), (xyz_points_norm >= 0).all(axis=2)).astype(np.uint8)
    mask = np.tile(mask.reshape(xyz_points_norm.shape[0], xyz_points_norm.shape[1], 1), (1, 1, 3))
    xyz_points_norm = xyz_points_norm * mask

    return xyz_points_norm


def plot_xyz(xyz_points):

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


def report_xyz_stats(xyz_points):

    # remove outliers and noise from the depth image - e.g. 0s
    np.place(xyz_points[...,2], xyz_points[...,2] == 0, np.mean(xyz_points[...,2]))

    print("Min X: {0}".format(np.min(xyz_points[...,0])))
    print("Max X: {0}".format(np.max(xyz_points[...,0])))
    print("Min Y: {0}".format(np.min(xyz_points[...,1])))
    print("Max Y: {0}".format(np.max(xyz_points[...,1])))
    print("Min Z: {0}".format(np.min(xyz_points[...,2])))
    print("Max Z: {0}".format(np.max(xyz_points[...,2])))
    print("# of 0s: ".format(np.product(xyz_points.shape[:-1]) - np.count_nonzero(xyz_points[...,2])))

    x_norm = (xyz_points[...,0] - np.min(xyz_points[...,0])) / (np.max(xyz_points[...,0]) - np.min(xyz_points[...,0]))
    y_norm = (xyz_points[...,1] - np.min(xyz_points[...,1])) / (np.max(xyz_points[...,1]) - np.min(xyz_points[...,1]))
    z_norm = (xyz_points[...,2] - np.min(xyz_points[...,2])) / (np.max(xyz_points[...,2]) - np.min(xyz_points[...,2]))

    
    bins = 100
    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    axs[0].hist(x_norm.flatten(), bins=bins)
    axs[0].set_title("X")
    axs[1].hist(y_norm.flatten(), bins=bins)
    axs[1].set_title("Y")
    axs[2].hist(z_norm.flatten(), bins=bins)
    axs[2].set_title("Z")
    plt.show()
    plt.close()

    cv2.imshow("X", x_norm)
    cv2.imshow("Y", y_norm)
    cv2.imshow("Z", z_norm)
    cv2.waitKey(0)


#################
### Core Func ###
#################


class Kinect_Data_Processor(object):

    def __init__(self, debug=False, parameters=None):
        self.debug = debug
        self.output = []
        self.counter = 0
        self.parameters = parameters
        self.bounds = {'x':[0.1, 1.3], 'y':[-0.5, 0.5], 'z':[0.0, 1.0]}

        # hue ranges - [0,180] - used when classifiying patches wrt color labels
        self.hues = {}
        self.hues['blue'] = [90, 130]
        self.hues['yellow'] = [5, 30]
        self.hues['green'] = [30, 65]
        self.hues['red'] = [170, 180]
        self.hues['purple'] = [130, 160]


    def callback(self, data):

        self.counter += 1
        print(self.counter)

        cloud = pointcloud2_to_array(data, split_rgb=True, remove_padding=True)
        xyz, bgr = get_xyzbgr_points(cloud, remove_nans=False)
        xyz = np.nan_to_num(xyz)

        # max height - 424
        # max width  - 512
        height = 540
        width = 960
        offset = 0
        margin_h = 50
        margin_w = 100
        # half of the crop's size; we assume a square crop
        crop_size = 60

        image_bbox = [offset + margin_h, offset + height - margin_h, offset + margin_w, offset + width - margin_w]

        xyz = xyz[image_bbox[0] : image_bbox[1], image_bbox[2] : image_bbox[3], :]
        bgr = bgr[image_bbox[0] : image_bbox[1], image_bbox[2] : image_bbox[3], :]

        height -= 2*margin_h
        width -= 2*margin_w

        # filter only the carpet parts of the point cloud'

        image = bgr.copy()
        bg_patch_size = 20
        image_size = image.shape[:-1]
        number_of_tiles = (int(image_size[0] / bg_patch_size), int(image_size[1] / bg_patch_size), 1)
        bg_threshold = 50
        
        backgrounds = []
        masks = []
        
        # take patches for bg from the 4 courners
        backgrounds.append(np.tile(image[0 : bg_patch_size, \
                                   0 : bg_patch_size, :], \
                                   number_of_tiles))
        backgrounds.append(np.tile(image[image_size[0] - bg_patch_size : image_size[0], \
                                   0 : bg_patch_size, :], \
                                   number_of_tiles))
        backgrounds.append(np.tile(image[0 : bg_patch_size, \
                                   image_size[1] - bg_patch_size : image_size[1], :], \
                                   number_of_tiles))
        backgrounds.append(np.tile(image[image_size[0] - bg_patch_size : image_size[0], \
                                   image_size[1] - bg_patch_size : image_size[1], :], \
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
        mask = np.logical_and(np.any(bgr < 150, axis=2), np.any(bgr > 0, axis=2)).astype(np.uint8)
        # cv2.imshow("Pure Mask", mask * 255)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # cv2.imshow("Smoothed", mask * 255)
        # cv2.imshow("bgr", bgr * np.tile(mask[:,:,np.newaxis], (1,1,3)))
        # cv2.waitKey(1)

        mask_copy = copy.deepcopy(mask)
        contours, hier = cv2.findContours(mask_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cnt) for cnt in contours]
        contours = [x for _,x in sorted(zip(areas,contours), reverse=True)]
        
        # areas = [x for x,_ in sorted(zip(areas,contours), reverse=True)]
        # print(len(contours))
        # print(areas)

        # return(0)
        
        # extract the color and spatial information for each object
        new_mask = np.zeros((height, width))
        new_entry = [[],[]]
        for cnt in contours[:self.parameters['no_objects']]:
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
            xyz_crop_trans = transform_xyz(xyz_crop, target_frame="base_link", height=crop_size * 2, width=crop_size * 2)
            xyz_crop_trans_norm = normalise_xyz(xyz_crop_trans, bounds=self.bounds)
            bgr_crop = bgr[bbox[0] : bbox[1], bbox[2] : bbox[3], :]

            result = {}

            # filter out the table artefacts
            xyz_crop_trans_norm[(bgr_crop > 170).all(axis=2)] = 0
            bgr_crop[(bgr_crop > 170).all(axis=2)] = 0

            for hue in self.hues:
                hsv_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
                xyz = xyz_crop_trans_norm[np.logical_and(hsv_crop[...,0] >= self.hues[hue][0], hsv_crop[...,0] <= self.hues[hue][1])]

                if len(xyz) < 200:
                    continue

                # result[hue] = xyz_crop_trans_norm[np.logical_and(hsv_crop[...,0] >= hues[hue][0], hsv_crop[...,0] <= hues[hue][1])]
                # hsv_crop[np.logical_or(hsv_crop[...,0] < hues[hue][0], hsv_crop[...,0] > hues[hue][1])] = 0

                # cv2.imshow("crop", cv2.cvtColor(hsv_crop, cv2.COLOR_HSV2BGR))
                # cv2.waitKey(0)

            # np.savez("/home/yordan/pr2_ws/src/spatial_relations_experiments/kinect_processor/cloud.npz", **result)
            # exit()

                assert ((xyz <= 1).all() and (xyz >= 0).all()),\
                "The data can not be normalised in the range [0,1] - Potentially bad bounds"

                new_entry[0].append(xyz)
                new_entry[1].append(hue)

                # if self.debug:
                #     # build up a mask with regions of interest
                #     new_mask[bbox[0] : bbox[1], bbox[2] : bbox[3]] = 1
        
        self.output.append(new_entry)

        # if self.debug:
        #     xyz_trans = transform_xyz(xyz, target_frame="base_link", height=height, width=width)
        #     xyz_trans_norm = normalise_xyz(xyz_trans, bounds=self.bounds)

        #     # extend the mask to have 3 channels - for RGB/XYZ - and filter the images
        #     mask = new_mask
        #     mask = np.tile(mask.reshape(height, width, 1), (1, 1, 3))
        #     bgr_fil = (bgr * mask).astype(np.uint8)
        #     xyz_fil_trans_norm = (xyz_trans_norm * mask)

        #     # print(xyz_crop_trans_norm)
        #     plot_xyz(xyz_crop_trans_norm)
        #     # bad_mask_tmp = (xyz_fil_trans_norm > 1.1).any(axis=2).astype(np.uint8)
        #     # bad_mask = np.tile(bad_mask_tmp.reshape(height, width, 1), (1,1,3))

        #     cv2.imshow("Mask", (mask * 255).astype(np.uint8))
        #     # cv2.imshow("Bad Mask", (bad_mask * 255).astype(np.uint8))
        #     cv2.imshow("BGR", bgr)
        #     cv2.imshow("BGR Filtered", bgr_fil)
        #     cv2.waitKey(10)

    def fill_holes_get_max_cnt(self, mask):
        """Given a binary mask, fills in any holes in the contours, selects the contour with max area
        and returns a mask with only it drawn + its bounding box
        """
        canvas = np.zeros(mask.shape, dtype=np.uint8)
        cnts, hier = cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cnt) for cnt in cnts]

        (cnt, area) = sorted(zip(cnts, areas), key=lambda x: x[1], reverse=True)[0]
        cv2.drawContours(canvas,[cnt],0,1,-1)

        return canvas.astype(np.uint8)


    def save_to_npz(self, output_folder):

        np.savez(os.path.join(output_folder, "rosbag_dump.npz"), self.output)


if __name__ == '__main__':

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('kinect_processor', anonymous=True)

    parameters = rospy.get_param("/kinect_processor")

    k_processor = Kinect_Data_Processor(debug=True, parameters=parameters)
    rospy.Subscriber("/kinect2/qhd/points", PointCloud2, k_processor.callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

    print("Final count: "+ str(k_processor.counter))
    k_processor.output = np.array(k_processor.output)
    k_processor.save_to_npz("/home/yordan/pr2_ws/src/spatial_relations_experiments/kinect_processor/rosbag_dump")
    print("NPZ saved")
