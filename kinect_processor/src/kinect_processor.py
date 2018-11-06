#!/usr/bin/env python
"""
title           :kinect_processor.py
description     :Processes raw sensory data captured from a Kinect 2 under the ROS framework. Given
                :user instructions, saves crops from the sensory feed in a numpy array under 
                :learning_experiments/data/.
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

    def __init__(self, debug=False):
        self.debug = debug
        self.output = []
        self.counter = 0
        self.bounds = {'x':[0.1, 1.3], 'y':[-0.5, 0.5], 'z':[0.0, 1.0]}


    def callback(self, data):

        self.counter += 1

        cloud = pointcloud2_to_array(data, split_rgb=True, remove_padding=True)
        xyz, bgr = get_xyzbgr_points(cloud, remove_nans=False)
        xyz = np.nan_to_num(xyz)

        # max height - 424
        # max width  - 512
        height = 424
        width = 512
        offset = 0
        margin = 50
        # half of the crop's size; we assume a square crop
        crop_size = 14
        xyz = xyz[offset + margin : offset + height - margin, offset + margin : offset + width - margin, :]
        bgr = bgr[offset + margin : offset + height - margin, offset + margin : offset + width - margin, :]

        height -= 2*margin
        width -= 2*margin

        # filter only the 'non-white parts of the point cloud'
        mask = np.any(np.logical_and(bgr < 50, bgr > 0), axis=2).astype(np.uint8)
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        mask_copy = copy.deepcopy(mask)
        contours,hier = cv2.findContours(mask_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cnt) for cnt in contours]
        contours = [x for _,x in sorted(zip(areas,contours), reverse=True)]
        
        # extract the color and spatial information for each object
        new_mask = np.zeros((height, width))
        new_entry = [[],[]]
        for cnt in contours[:2]:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            xyz_crop = xyz[cY - crop_size : cY + crop_size, cX - crop_size : cX + crop_size, :]
            xyz_crop_trans = transform_xyz(xyz_crop, target_frame="base_link", height=crop_size * 2, width=crop_size * 2)
            xyz_crop_trans_norm = normalise_xyz(xyz_crop_trans, bounds=self.bounds)
            bgr_crop = bgr[cY - crop_size : cY + crop_size, cX - crop_size : cX + crop_size, :]

            assert ((xyz_crop_trans_norm <= 1).any() and (xyz_crop_trans_norm >= 0).any()),\
            "The data can not be normalised in the range [0,1] - Potentially bad bounds"

            new_entry[0].append(xyz_crop_trans_norm)
            new_entry[1].append(bgr_crop)

            if self.debug:
                # build up a mask with regions of interest
                new_mask[cY - crop_size : cY + crop_size, cX - crop_size : cX + crop_size] = 1
        
        self.output.append(new_entry)

        if self.debug:
            xyz_trans = transform_xyz(xyz, target_frame="base_link", height=height, width=width)
            xyz_trans_norm = normalise_xyz(xyz_trans, bounds=self.bounds)

            # extend the mask to have 3 channels - for RGB/XYZ - and filter the images
            mask = new_mask
            mask = np.tile(mask.reshape(height, width, 1), (1, 1, 3))
            bgr_fil = (bgr * mask).astype(np.uint8)
            xyz_fil_trans_norm = (xyz_trans_norm * mask)

            # plot_xyz(xyz_fil_transformed_norm)
            # bad_mask_tmp = (xyz_fil_transformed_norm > 1.1).any(axis=2).astype(np.uint8)
            # bad_mask = np.tile(bad_mask_tmp.reshape(height, width, 1), (1,1,3))

            cv2.imshow("Mask", (mask * 255).astype(np.uint8))
            # cv2.imshow("Bad Mask", (bad_mask * 255).astype(np.uint8))
            cv2.imshow("BGR", bgr)
            cv2.imshow("BGR Filtered", bgr_fil)
            cv2.waitKey(10)

class Data_Annotator(object):

    def __init__(self, parameters):

        # used when classifiying patches wrt color labels
        self.hues = {}
        self.hues['blue'] = [90, 130]
        self.hues['yellow'] = [5, 35]
        self.hues['green'] = [35, 65]
        self.hues['red'] = [160, 180]

        # uses the information from the instructions to construct the conceptial groups
        self.data = {}
            
        for i in range(len([x for x in parameters if 'instruction ' in x])):
            instruction = parameters['instruction ' + str(i)]
            labels = instruction.split()[1].split('/')

            array_key = '_'.join(labels) + '_' + '_'.join(instruction.split()[:1] + instruction.split()[2:])
            # contains two arrays for the inputs of the two network branches
            self.data[array_key] = [[],[]]

        print(self.data)


    # checks whether a given color patch can be 
    def is_color(self, bgr, color):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) 
        lower = np.array([self.hues[color][0],50,50]) 
        upper = np.array([self.hues[color][1], 255, 255]) 

        mean_hsv = np.mean(hsv, axis=(0,1))
        # print(color, mean_hsv)

        return all(mean_hsv >= lower) and all(mean_hsv <= upper)


    # given the linguistic instruction, decide which crop goes to which branch of the net and what
    # spatial label is the tuple given
    def annotate(self, data, debug=False):
        
        no_input_pairs = len(data)

        for entry in data:
            xyz_crops = entry[0]
            bgr_crops = entry[1]
            bgr_out = [[],[]]

            for array_key in self.data:

                # add the crops to the corresponding array, given the instructions
                object_labels = array_key.split('_')[2:]
                
                for bgr_index, bgr_crop in enumerate(bgr_crops):

                    for index, object_label in enumerate(object_labels):
                        if self.is_color(bgr_crop.astype(np.uint8), object_label):
                            # print(object_label)
                            self.data[array_key][index].append(xyz_crops[bgr_index])
                            bgr_out[index] = bgr_crop.astype(np.uint8)
                            break
        
                if debug:
                    print(bgr_out[0])
                    cv2.imshow("first", bgr_out[0])
                    print(self.data[2][-1])
                    print(self.data[3][-1])
                    print('\n')
                    cv2.waitKey(0)

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


    def save_to_npz(self):

        for key in self.data.keys():
            output = {"branch_0":self.data[key][0], "branch_1":self.data[key][1]}
            np.savez(os.path.join("/home/yordan/pr2_ws/src/spatial_relations_experiments/learning_experiments/data/train/", key + ".npz"), **output)

def main_loop():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('kinect_processor', anonymous=True)

    k_processor = Kinect_Data_Processor(debug=True)
    rospy.Subscriber("/kinect2/sd/points", PointCloud2, k_processor.callback)

    parameters = rospy.get_param("/kinect_processor")

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

    print("Final count: "+ str(k_processor.counter))
    k_processor.output = np.array(k_processor.output)
    print(k_processor.output.shape)

    annotator = Data_Annotator(parameters)
    annotator.annotate(k_processor.output)
    annotator.save_to_npz()
    print("NPZ saved")

if __name__ == '__main__':

    main_loop()


    # rospy.init_node('kinect_listener')
    # tf_listener = tf.TransformListener()
    # transformer = tf.Transformer(True, rospy.Duration(10.0))
    # tf_listener.waitForTransform('/base_link','/head_mount_kinect2_ir_optical_frame',rospy.Time(), rospy.Duration(4.0))

    # rate = rospy.Rate(10.0)
    # while not rospy.is_shutdown():
    #     tf_listener.waitForTransform('/base_link','/head_mount_kinect2_ir_optical_frame',rospy.Time(), rospy.Duration(4.0))
    #     position, quaternion = tf_listener.lookupTransform("/base_link", "/head_mount_kinect2_ir_optical_frame", rospy.Duration(0))
    #     print position, quaternion
    #     rate.sleep()
