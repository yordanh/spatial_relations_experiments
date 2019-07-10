"""
title           :data_generator_robot_data.py
description     :Loads the spatial dataset contained in numpy arrays under train,unseen,ulabelled 
                :folders under learning_experiments/data/.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :10/2018
python_version  :2.7.6
==============================================================================
"""

import os
import os.path as osp
import cv2
import numpy as np
import json
import shutil
import glob
import yaml
import pprint
from tqdm import tqdm

import chainer
# import chainer_mask_rcnn as cmr

# remove the following imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from chainer.datasets import TupleDataset

class DataGenerator(object):
    def __init__(self, label_mode=None, augment_flag=True, folder_names=["yordan_experiments"], data_split=0.8, include_eef=False):
        self.label_mode = label_mode
        self.augment_flag = augment_flag
        self.folder_names = folder_names
        self.data_split = data_split
        self.mask_mode = 'loading'
        self.include_eef = include_eef

        self.mask_rcnn = None


    def segment_mask(self, bgr, expected_labels):
        
        batch = np.array(np.transpose(bgr, (2,0,1)))
        batch = batch[np.newaxis, :]
        bboxes, masks, labels, scores = self.mask_rcnn.predict(batch)

        indecies = scores[0] >= self.score_threshold
        bboxes = bboxes[0][indecies]
        masks = masks[0][indecies]
        labels = labels[0][indecies]
        labels = self.class_names[labels]
        scores = scores[0][indecies]

        if len(masks != 0):
            tuples = [(i, label, score) for i, (label, score) in enumerate(zip(labels, scores))]

            good_indecies = []
            for expected_label in expected_labels:
                filtered = [(i, label, score) for (i, label, score) in tuples if label == expected_label]
                try:
                    good_indecies.append(sorted(filtered, key=lambda x: x[2], reverse=True)[0][0])
                except:
                    print("BAD; a mask for {0} is missing".format(expected_label))

            masks = np.take(masks, good_indecies, axis=0)
            labels = np.take(labels, good_indecies)
            scores = np.take(scores, good_indecies)

            for i in range(len(masks)):
                if labels[i] in expected_labels:
                    masks[i] = self.fill_holes_get_max_cnt(masks[i])

        return masks, labels, scores


    def fill_holes_get_max_cnt(self, mask, fill=-1):
        """Given a binary mask, fills in any holes in the contours, selects the contour with max area
        and returns a mask with only it drawn + its bounding box
        """

        # cv2.imshow("mask orig", (mask*255).astype(np.uint8))

        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

        canvas = np.zeros(mask.shape, dtype=np.uint8)
        cnts, hier = cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cnt) for cnt in cnts]

        (cnt, area) = sorted(zip(cnts, areas), key=lambda x: x[1], reverse=True)[0]
        cv2.drawContours(canvas,[cnt],0,1,fill)

        mask = canvas.astype(np.uint8)

        # cv2.imshow("mask", mask*255)
        # cv2.waitKey()

        return mask.astype(np.uint8)


    def load_model(self, folder_name="./maskrcnn_model", gpu_id=0):

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # param
        params = yaml.load(open(osp.join(folder_name, 'params.yaml')))
        # print('Training config:')
        # print('# ' + '-' * 77)
        # pprint.pprint(params)
        # print('# ' + '-' * 77)

        # dataset
        if 'class_names' in params:
            class_names = params['class_names']
        else:
            raise ValueError

        # model

        if params['dataset'] == 'voc':
            if 'min_size' not in params:
                params['min_size'] = 600
            if 'max_size' not in params:
                params['max_size'] = 1000
            if 'anchor_scales' not in params:
                params['anchor_scales'] = (1, 2, 4, 8, 16, 32)
        elif params['dataset'] == 'coco':
            if 'min_size' not in params:
                params['min_size'] = 800
            if 'max_size' not in params:
                params['max_size'] = 1333
            if 'anchor_scales' not in params:
                params['anchor_scales'] = (1, 2, 4, 8, 16, 32)
        else:
            assert 'min_size' in params
            assert 'max_size' in params
            assert 'anchor_scales' in params

        if params['pooling_func'] == 'align':
            pooling_func = cmr.functions.roi_align_2d
        elif params['pooling_func'] == 'pooling':
            pooling_func = cmr.functions.roi_pooling_2d
        elif params['pooling_func'] == 'resize':
            pooling_func = cmr.functions.crop_and_resize
        else:
            raise ValueError(
                'Unsupported pooling_func: {}'.format(params['pooling_func'])
            )

        model_name = [x for x in os.listdir(folder_name) if ".npz" in x][0]
        pretrained_model = osp.join(folder_name, model_name)
        print('Using pretrained_model: %s' % pretrained_model)

        model = params['model']
        self.mask_rcnn = cmr.models.MaskRCNNResNet(
            n_layers=int(model.lstrip('resnet')),
            n_fg_class=len(class_names),
            pretrained_model=pretrained_model,
            pooling_func=pooling_func,
            anchor_scales=params['anchor_scales'],
            mean=params.get('mean', (123.152, 115.903, 103.063)),
            min_size=params['min_size'],
            max_size=params['max_size'],
            roi_size=params.get('roi_size', 7),
        )
        
        self.class_names = np.array(class_names)
        self.score_threshold = 0.05

        # self.mask_rcnn.to_cpu()
        chainer.cuda.get_device_from_id(gpu_id).use()
        self.mask_rcnn.to_gpu()


    def quaternion_to_euler(self, quat):

        (x,y,z,w) = quat / np.linalg.norm(quat)
        
        import math
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1)) + 180
        X /= (360.)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2)) + 180
        Y /= (360.)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4)) + 180
        Z /= (360.)

        return X, Y, Z


    def generate_dataset(self, ignore=["unlabelled"], args=None):
        
        crop_size = 128
        data_dimensions = [crop_size, crop_size, 7]

        seed = 0
        np.random.seed(seed)
        
        possible_groups = [['off', 'on'], 
                           ['nonfacing', 'facing'],
                           ['out', 'in']]

        object_colors = ['red', 'blue', 'yellow', 'purple']
        object_shapes = ['cube', 'cup', 'bowl']

        # self.groups_rel = {0 : possible_groups[0]}
        #                1 : possible_groups[1],
        #                2 : possible_groups[2]}
        self.groups_rel = {i : possible_groups[i] for i in range(len(possible_groups))}

        self.groups_obj = {0 : object_colors,
                       1 : object_shapes}

        self.cutoff_for_labels = 30
        
        print(self.groups_rel)
        print(self.groups_obj)

        expected_labels_list = [['purple_cup', 'red_cube'], ['purple_cup', 'blue_cup'], ['purple_bowl', 'yellow_cube']]
        # expected_labels_list = [['purple_cup', 'blue_cup'], ['purple_bowl', 'yellow_cube']]
        relationships_start = []
        relationships = {'off':[[1],[]], 
                         'on':[[],[]], 
                         'nonfacing':[[],[]], 
                         'facing':[[],[]], 
                         'out':[[],[]], 
                         'in':[[],[]]}
        relationships_start.append(relationships)

        relationships = {'off':[[],[]], 
                         'on':[[],[]], 
                         'nonfacing':[[1],[]], 
                         'facing':[[],[]], 
                         'out':[[],[]], 
                         'in':[[],[]]}
        relationships_start.append(relationships)

        relationships = {'off':[[],[]], 
                         'on':[[],[]], 
                         'nonfacing':[[],[]], 
                         'facing':[[],[]], 
                         'out':[[1],[]], 
                         'in':[[],[]]}
        relationships_start.append(relationships)

        relationships_end = []
        relationships = {'off':[[],[]], 
                         'on':[[1],[]], 
                         'nonfacing':[[],[]], 
                         'facing':[[],[]], 
                         'out':[[],[]], 
                         'in':[[],[]]}
        relationships_end.append(relationships)

        relationships = {'off':[[],[]], 
                         'on':[[],[]], 
                         'nonfacing':[[],[]], 
                         'facing':[[1],[]], 
                         'out':[[],[]], 
                         'in':[[],[]]}
        relationships_end.append(relationships)

        relationships = {'off':[[],[]], 
                         'on':[[],[]], 
                         'nonfacing':[[],[]], 
                         'facing':[[],[]], 
                         'out':[[],[]], 
                         'in':[[1],[]]}
        relationships_end.append(relationships)

        scene_objs_all = []
        scene_objs_all.append([{'color' : 'purple', 'shape' : 'cup'},
                               {'color' : 'red', 'shape' : 'cube'}])
        scene_objs_all.append([{'color' : 'purple', 'shape' : 'cup'},
                               {'color' : 'blue', 'shape' : 'cup'}])
        scene_objs_all.append([{'color' : 'purple', 'shape' : 'bowl'},
                               {'color' : 'yellow', 'shape' : 'cube'}])

        train = []
        train_labels = []
        train_vectors = []
        train_masks = []
        train_object_vectors = []
        train_object_vector_masks = []
        train_eefs = []
        
        test = []
        test_labels = []
        test_vectors = []
        test_masks = []
        test_object_vectors = []
        test_object_vector_masks = []
        test_eefs = []

        unseen = []
        unseen_labels = []
        unseen_vectors = []
        unseen_masks = []
        unseen_object_vectors = []
        unseen_object_vector_masks = []

        for folder_name in self.folder_names[:]:

            demonstrations = sorted(os.listdir(folder_name))

            # if "_2" not in folder_name and "_3" not in folder_name:
            #     demonstrations = demonstrations[:]
            # else:
            #     demonstrations = demonstrations[:][:-5]

            for demonstration in demonstrations[:]:
            # for demonstration in demonstrations[:10]:
                # print(len(train))
                # print(len(test))
                files = glob.glob(osp.join(folder_name, demonstration, "kinect2_qhd_image_color*.jpg"))
                files = sorted([x.split('/')[-1].replace('kinect2_qhd_image_color_rect_', '').replace('.jpg', '') for x in files])

                if osp.exists(osp.join(folder_name, demonstration, 'segmented_masks.npy')):
                    self.mask_mode = 'loading'
                    masks_loaded_array = np.load(osp.join(folder_name, demonstration, 'segmented_masks.npy'))
                else:
                    self.mask_mode = 'segmenting'
                    masks_output_array = {}

                    if self.mask_rcnn == None:
                        # import chainer_mask_rcnn as cmr
                        self.load_model(gpu_id=0)

                # file_list_train = files[:10] + files[-20:]
                file_list_train = files[:]

                number_of_files = len(file_list_train)
                train_n = int(self.data_split * number_of_files)
                test_n = number_of_files - train_n
            
                train_indecies = np.random.choice(range(number_of_files), train_n, replace=False)
                test_indecies = np.array(filter(lambda x : x not in train_indecies, range(number_of_files)))

                train_files = np.take(file_list_train, train_indecies)
                test_files = np.take(file_list_train, test_indecies)

                if "train" not in ignore:
                    for file_idx in tqdm(range(len(file_list_train))):

                        if file_idx > self.cutoff_for_labels and len(file_list_train) - file_idx > self.cutoff_for_labels:
                            if file_idx % 3 != 0:
                                continue

                        file_name = file_list_train[file_idx]

                        if (file_list_train.index(file_name)) == 0:
                            print("Processing FOLDER {0}, {1}/{2}".format(folder_name, 
                                                                          demonstrations.index(demonstration) + 1, 
                                                                          len(demonstrations)))

                        orig_dims = [540, 960]

                        if "_2" not in folder_name and "_3" not in folder_name:
                            desired_dim = 384
                        else:
                            desired_dim = 512

                        if "_3" in folder_name and 'facing' in folder_name:
                            desired_dim = 384

                        crop_window = [[], []]
                        crop_window[0].append(orig_dims[0]/2 - desired_dim/2)
                        crop_window[0].append(orig_dims[0]/2 + desired_dim/2)
                        crop_window[1].append(orig_dims[1]/2 - desired_dim/2)
                        crop_window[1].append(orig_dims[1]/2 + desired_dim/2)

                        bgr = cv2.imread(osp.join(folder_name, demonstration, 
                                         "kinect2_qhd_image_color_rect_" + file_name + ".jpg"))
                        bgr = bgr[crop_window[0][0] : crop_window[0][1], 
                                  crop_window[1][0] : crop_window[1][1], 
                                  :]
                        bgr = bgr / 255.


                        # if file_idx < self.cutoff_for_labels or len(file_list_train) - file_idx < self.cutoff_for_labels:
                        #     cv2.imshow("bgr", (bgr * 255).astype(np.uint8))
                        #     cv2.waitKey(50)
                        #     continue

                        depth_orig = cv2.imread(osp.join(folder_name, demonstration, 
                                                "kinect2_qhd_image_depth_rect_" + file_name + ".jpg"))
                        depth_orig = depth_orig[crop_window[0][0] : crop_window[0][1], 
                                                crop_window[1][0] : crop_window[1][1], 
                                                :]
                        # depth_orig = depth_orig / 255.
                        depth_orig = depth_orig / float(np.max(depth_orig))
                        
                        if self.include_eef:
                            eef_pose = []

                            if file_idx < self.cutoff_for_labels:
                                eef_file_name = file_list_train[-file_idx]
                            elif len(file_list_train) - file_idx < self.cutoff_for_labels:
                                eef_file_name = file_list_train[len(file_list_train) - file_idx]
                            else:
                                eef_file_name = file_name

                            f = open(osp.join(folder_name, demonstration, 
                                     "r_wrist_roll_link_" + file_name + '.txt'))
                            for line in f:    
                                eef_pose.append(float(line))

                            r, p, yaw = self.quaternion_to_euler(eef_pose[-4:])
                            
                            x = eef_pose[0]
                            y = eef_pose[1] + 0.5
                            z = eef_pose[2]

                            if self.augment_flag:

                                dist_range = 0
                                r += np.random.uniform(-dist_range, dist_range, size=1)
                                p += np.random.uniform(-dist_range, dist_range, size=1)
                                yaw += np.random.uniform(-dist_range, dist_range, size=1)

                                dist_range = 0
                                x += np.random.uniform(-dist_range, dist_range, size=1)
                                y += np.random.uniform(-dist_range, dist_range, size=1)
                                z += np.random.uniform(-dist_range, dist_range, size=1)

                                eef_pose = [x[0], y[0], z[0], r[0], p[0], yaw[0]]

                            else:
                                eef_pose = [x, y, z, r, p, yaw]

                            # print(eef_pose)

                        if self.mask_mode == 'loading':
                            dict_entry = masks_loaded_array.item().get(file_name)

                            if dict_entry == None:
                                print("BAD")
                                continue

                            masks = dict_entry.values()
                            labels = dict_entry.keys()
                            scores = np.zeros(len(masks))

                            # if file_idx < self.cutoff_for_labels or len(file_list_train) - file_idx < self.cutoff_for_labels:
                            #     cv2.imshow("bgr", (bgr * 255).astype(np.uint8))
                            #     cv2.imshow("depth", (depth_orig * 255).astype(np.uint8))

                            #     for mask_idx, mask in enumerate(masks):
                            #         cv2.imshow("mask " + labels[mask_idx], (mask * 255).astype(np.uint8))

                            #     cv2.waitKey(50)

                        elif self.mask_mode == 'segmenting':
                            expected_labels = expected_labels_list[self.folder_names.index(folder_name)]
                            masks, labels, scores = self.segment_mask((bgr * 255).astype(np.uint8), expected_labels)
                            
                            if len(masks) != 2:
                                continue

                            masks_output_array[file_name] = {label : mask for label,mask in zip(labels, masks)}

                            continue



                        n_obj = 2
                        init_rel = np.array(['unlabelled' for _ in range(3)])
                        rel_index = {x : {y : init_rel.copy() for y in np.delete(np.arange(n_obj), x)} for x in np.arange(n_obj)}

                        if file_idx < self.cutoff_for_labels:
                            rels = relationships_end[self.folder_names.index(folder_name)]
                        elif len(file_list_train) - file_idx < self.cutoff_for_labels:
                            rels = relationships_start[self.folder_names.index(folder_name)]
                        else:
                            rels = {'off':[[],[]], 
                                    'on':[[],[]], 
                                    'nonfacing':[[],[]], 
                                    'facing':[[],[]], 
                                    'out':[[],[]], 
                                    'in':[[],[]]}

                        scene_objs = scene_objs_all[self.folder_names.index(folder_name)]

                        for rel_name, obj_list in rels.items():
                            for i in self.groups_rel:
                                if rel_name in self.groups_rel[i]:
                                    group_idx = i
                            for ref_idx, target_list in enumerate(obj_list):
                                for target_idx in target_list:
                                    rel_index[ref_idx][target_idx][group_idx] = rel_name


                        for (ref_idx, target_list) in rel_index.items():
                            for (target_idx, rel_list) in target_list.items():
                                
                                # scale = 0.125
                                # scale = 1
                                if "_2" not in folder_name and "_3" not in folder_name:
                                    scale = 0.3333333333333
                                else:
                                    scale = 0.25

                                if "_3" in folder_name and 'facing' in folder_name:
                                    scale = 0.3333333333333

                                color = cv2.resize(bgr.copy(), (0,0), fx=scale, fy=scale)
                                depth = cv2.resize(depth_orig.copy(), (0,0), fx=scale, fy=scale)
                                bg = np.zeros((depth.shape[0], depth.shape[1]))
                                ref = cv2.resize(masks[ref_idx].copy().astype(np.uint8), (0,0), fx=scale, fy=scale)
                                ref = ref.astype(np.float32)
                                tar = cv2.resize(masks[target_idx].copy().astype(np.uint8), (0,0), fx=scale, fy=scale)
                                tar = tar.astype(np.float32)

                                # print(color.shape)
                                # print(depth.shape)
                                # print(ref.shape)
                                # print(tar.shape)

                                if np.sum(tar) == 0 or np.sum(ref) == 0 or (ref == tar).all():
                                    continue

                                pixels = np.concatenate((color, depth[...,0,None], bg[...,None], ref[...,None], tar[...,None]), axis=2)

                                if file_name in train_files:
                                    
                                    train += [pixels]
                                    train_labels.append(rel_list)
                                    mask = []
                                    vector = []

                                    for i, rel_name in enumerate(rel_list):

                                        if rel_name != "unlabelled":
                                            vector.append(self.groups_rel[i].index(rel_name))
                                            mask.append(1)
                                        else:
                                            vector.append(0)
                                            mask.append(0)

                                    train_masks.append(mask)
                                    train_vectors.append(vector)

                                    train_object_vectors.append([])
                                    train_object_vector_masks.append([])

                                    for idx in [ref_idx, target_idx]:
                                        color = scene_objs[idx]['color']
                                        shape = scene_objs[idx]['shape']
                                        train_object_vectors[-1].append([object_colors.index(color),\
                                                                 object_shapes.index(shape)])
                                        train_object_vector_masks[-1].append([1, 1])
                                    
                                    if self.include_eef:
                                        train_eefs.append(eef_pose)
                                
                                elif file_name in test_files:
                                    
                                    test += [pixels]
                                    test_labels.append(rel_list)
                                    mask = []
                                    vector = []

                                    for i, rel_name in enumerate(rel_list):

                                        if rel_name != "unlabelled":
                                            vector.append(self.groups_rel[i].index(rel_name))
                                            mask.append(1)
                                        else:
                                            vector.append(0)
                                            mask.append(0)
                                    
                                    test_masks.append(mask)
                                    test_vectors.append(vector)

                                    test_object_vectors.append([])
                                    test_object_vector_masks.append([])

                                    for idx in [ref_idx, target_idx]:
                                        color = scene_objs[idx]['color']
                                        shape = scene_objs[idx]['shape']
                                        test_object_vectors[-1].append([object_colors.index(color),\
                                                                 object_shapes.index(shape)])
                                        test_object_vector_masks[-1].append([1, 1])

                                    if self.include_eef:                                   
                                        test_eefs.append(eef_pose)

                    if self.mask_mode == 'segmenting':
                        path = osp.join(folder_name, demonstration, 'segmented_masks.npy')
                        np.save(path, masks_output_array)
                
        train = np.array(train, dtype=np.float32)
        train_labels = np.array(train_labels)
        train_vectors = np.array(train_vectors)
        train_masks = np.array(train_masks)
        train_object_vectors = np.array(train_object_vectors)
        train_object_vector_masks = np.array(train_object_vector_masks)
        train_eefs = np.array(train_eefs)

        test = np.array(test, dtype=np.float32)
        test_labels = np.array(test_labels)
        test_vectors = np.array(test_vectors)
        test_masks = np.array(test_masks)
        test_object_vectors = np.array(test_object_vectors)
        test_object_vector_masks = np.array(test_object_vector_masks)
        test_eefs = np.array(test_eefs)

        print(train.shape)
        print(test.shape)
        
        unseen = np.array(unseen, dtype=np.float32)
        unseen_labels = np.array(unseen_labels)
        unseen_vectors = np.array(unseen_vectors)
        unseen_masks = np.array(unseen_masks)
        unseen_object_vectors = np.array(unseen_object_vectors)
        unseen_object_vector_masks = np.array(unseen_object_vector_masks)

        train = train.reshape([len(train)] + data_dimensions)
        test = test.reshape([len(test)] + data_dimensions)
        # unseen = unseen.reshape([len(unseen)] + data_dimensions)
        train = np.swapaxes(train, 1 ,3)
        test = np.swapaxes(test, 1 ,3)
        if unseen != []:
            unseen = np.swapaxes(unseen, 2 ,4)

        if self.include_eef:
            train_concat = TupleDataset(train, train_vectors, train_masks, \
                                        train_object_vectors, train_object_vector_masks, train_eefs)
        else:    
            train_concat = TupleDataset(train, train_vectors, train_masks, \
                                        train_object_vectors, train_object_vector_masks)
        if self.include_eef:        
            test_concat = TupleDataset(test, test_vectors, test_masks, \
                                       test_object_vectors, test_object_vector_masks, test_eefs)
        else:    
            test_concat = TupleDataset(test, test_vectors, test_masks, \
                                       test_object_vectors, test_object_vector_masks)
        unseen_concat = TupleDataset(unseen, unseen_vectors, unseen_masks, \
                                   unseen_object_vectors, unseen_object_vector_masks)

        result = []
        result.append(train)
        result.append(train_labels)
        result.append(train_concat)
        result.append(train_vectors)

        result.append(test)
        result.append(test_labels)
        result.append(test_concat)
        result.append(test_vectors)

        result.append(unseen)
        result.append(unseen_labels)
        result.append(unseen_concat)
        result.append(unseen_vectors)

        result.append(self.groups_obj)
        result.append(self.groups_rel)


        # for i,x in enumerate(test_concat[:]):

        #     image = x[0]
        #     image = np.swapaxes(image, 0 ,2)

        #     bgr = image[...,:3]
        #     bg = image[...,4]
        #     mask_obj_ref = image[...,5]
        #     mask_obj_tar = image[...,6]

        #     mask = x[2]
        #     vector = x[1]

        #     object_vectors = x[3]
        #     object_vector_masks = x[4]

        #     print("Labels", list(test_labels[i]))

        #     # print("Masks", mask)
        #     # print("Vectors", vector)

        #     print("Object vectors", object_vectors)
        #     # print("Object vector masks", object_vector_masks)

        #     # cv2.imshow("bg", (bg*255).astype(np.uint8))

        #     cv2.imshow("ref", (mask_obj_ref*255).astype(np.uint8))
        #     cv2.imshow("tar", (mask_obj_tar*255).astype(np.uint8))
        #     cv2.imshow("bgr", (bgr*255).astype(np.uint8))
            
        #     # if (mask_obj_ref == mask_obj_tar).all():
        #     #     cv2.imshow("diff", (mask_obj_ref != mask_obj_tar).astype(np.uint8) * 255)
        #     cv2.waitKey()


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
    BASE_DIR = "yordan_experiments_3"
    # folder_names = [osp.join(BASE_DIR, 'off-on'), 
    #                 osp.join(BASE_DIR, 'nonfacing-facing'), 
    #                 osp.join(BASE_DIR, 'out-in')]
    folder_names = [osp.join(BASE_DIR, 'off-on'), 
                    osp.join(BASE_DIR, 'out-in')]
    # folder_names = ['yordan_experiments/nonfacing-facing', 'yordan_experiments/out-in']
    data_generator = DataGenerator(folder_names=folder_names)
    result = data_generator.generate_dataset()

    # folder_names = ['outputs_test/left-right_no_no/' + str(i) for i in range(0,5)]
    # data_generator = DataGenerator(folder_names=folder_names)
    # result = data_generator.generate_dataset(ignore=['train'])