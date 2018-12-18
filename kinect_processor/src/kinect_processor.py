#!/usr/bin/env python
"""
title           :kinect_processor.py
description     :Processes raw sensory data captured from a Kinect 2 under the ROS framework:
                :parses, transforms xyz points and saves the frames under 
                :rosbag_dump/processed_rosbag_dump.npz
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :11/2018
python_version  :2.7.6
==============================================================================
"""

# ROS dependencies
import rospy
import tf
from tf import transformations
from sensor_msgs.msg import PointCloud2, PointField

# Misc Dependencies
import argparse
import struct
import copy
import time
import numpy as np
import cv2
import os
import os.path as osp
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

    def __init__(self, debug=False, args=None):
        self.debug = debug
        self.output = []
        self.counter = 0
        self.cutoff = args.cutoff
        self.scene = args.scene


    def callback(self, data):

        self.counter += 1
        if self.counter > self.cutoff:
            self.save_to_npz()
            rospy.signal_shutdown('Quit')
        else:
            print("{0} Kinect 2 frames processed.".format(self.counter))

            cloud = pointcloud2_to_array(data, split_rgb=True, remove_padding=True)
            xyz, bgr = get_xyzbgr_points(cloud, remove_nans=False)
            xyz = np.nan_to_num(xyz)
            xyz_transformed = transform_xyz(xyz, height=540, width=960)

            self.output.append((xyz_transformed, bgr))


    def save_to_npz(self, output_folder="scenes"):

        scene_list = os.listdir(output_folder)
        if self.scene not in scene_list:
            os.mkdir(osp.join(output_folder, self.scene))

        npz_list = [x for x in os.listdir(osp.join(output_folder, self.scene)) if '.npz' in x and "segmented" not in x]
        if npz_list == []:
            index = 0
        else:
            index = sorted([int(x.replace('.npz', '')) for x in npz_list])[-1] + 1

        np.savez(os.path.join(output_folder, self.scene, str(index) + ".npz"), self.output)
        print("NPZ saved.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Save Kinect data to NPZ file')
    parser.add_argument('--cutoff', default=100, type=int,
                        help='Number of frames to be captured')
    parser.add_argument('--scene', '-sc', default='0',
                        help='Index for a scene/setup')
    args = parser.parse_args()

    time.sleep(3)

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('kinect_processor', anonymous=True)

    k_processor = Kinect_Data_Processor(debug=True, args=args)
    rospy.Subscriber("/kinect2/qhd/points", PointCloud2, k_processor.callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
