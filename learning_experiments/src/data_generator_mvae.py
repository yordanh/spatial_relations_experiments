"""
title           :data_generator_mvae.py
description     :Loads the spatial dataset contained in numpy arrays under train,unseen,ulabelled 
                :folders under learning_experiments/photoreal_data/.
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

from chainer.datasets import TupleDataset

class DataGenerator(object):
    def __init__(self, label_mode=None, augment_counter=0, folder_names=["clevr_data"], data_split=0.8):
        self.label_mode = label_mode
        self.augment_counter = augment_counter
        self.folder_names = folder_names
        self.data_split = data_split

    def generate_dataset(self, ignore=["unlabelled"], args=None):
        
        crop_size = 128
        data_dimensions = [crop_size, crop_size, 7]

        seed = 0
        np.random.seed(seed)

        possible_groups = [['front', 'behind'], 
                           ['left', 'right'], 
                           ['above', 'below'],
                           ["close", "far"],
                           ["on", "off"],
                           ['in', 'out']]

        object_colors = ['red', 'blue', 'yellow', 'green', 'gray']
        object_sizes = ['large', 'small']
        object_shapes = ['sphere', 'cube', 'cylinder', 'tray']

        self.groups_rel = {i : possible_groups[i] for i in range(len(possible_groups))}

        self.groups_obj = {0 : object_colors,
                       1 : object_shapes,
                       2 : object_sizes}

        train = []
        train_labels = []
        train_vectors = []
        train_masks = []
        train_object_vectors = []
        train_object_vector_masks = []
        
        test = []
        test_labels = []
        test_vectors = []
        test_masks = []
        test_object_vectors = []
        test_object_vector_masks = []

        unseen = []
        unseen_labels = []
        unseen_vectors = []
        unseen_masks = []
        unseen_object_vectors = []
        unseen_object_vector_masks = []

        for folder_name in self.folder_names:
            folder_name_train_arr = osp.join(folder_name, "train", "arr")
            folder_name_train_scenes = osp.join(folder_name, "train", "scenes")

            array_list_train = sorted(os.listdir(folder_name_train_arr))[:]

            number_of_pcs = len(array_list_train)
            train_n = int(self.data_split * number_of_pcs)
            test_n = number_of_pcs - train_n
        
            train_indecies = np.random.choice(range(number_of_pcs), train_n, replace=False)
            test_indecies = np.array(filter(lambda x : x not in train_indecies, range(number_of_pcs)))

            train_files = np.take(array_list_train, train_indecies)
            test_files = np.take(array_list_train, test_indecies)

            unseen.append([])
            unseen_labels.append([])
            unseen_vectors.append([])
            unseen_masks.append([])
            unseen_object_vectors.append([])
            unseen_object_vector_masks.append([])

            if "train" not in ignore:
                for array_name in array_list_train:

                    data = np.load(osp.join(folder_name_train_arr, array_name))['arr_0']
                    scene_file = open(osp.join(folder_name_train_scenes, array_name.replace('.npz', '.json')), "r")
                    json_data = json.load(scene_file)
                    scene_objs = json_data['objects']
                    rels = json_data['relationships']

                    if (array_list_train.index(array_name)) == 0:
                        print("Processing FOLDER {0}".format(folder_name))

                    bgr = (data[...,:3] * 255).astype(np.uint8)
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
                    bgr = bgr / 255.

                    depth_orig = data[...,3]
                    depth_orig = depth_orig / np.max(depth_orig)
                    
                    mask = data[...,4:]
                    mask = (mask * 255).astype(np.uint8)

                    object_masks = {}
                    obj_pixel_vals = []

                    big_mask_flag = False

                    overlapping_objects = []
                    for i, obj in enumerate(scene_objs):
                        mask_tmp = mask.copy()
                        if 'mask_color' in obj:
                            obj_pixel_val = (np.array(obj['mask_color']) * 255).astype(np.uint8)
                        else:
                            pixel_coords = obj['pixel_coords'][:2]
                            pixel_coords = [127 if x > 127 else x for x in pixel_coords]
                            pixel_coords = [0 if x < 0 else x for x in pixel_coords]

                            obj_pixel_val = mask_tmp[pixel_coords[1], pixel_coords[0]]

                        obj_pixel_vals.append(list(obj_pixel_val))

                        object_masks[i] = (mask_tmp == obj_pixel_val).all(axis=2).astype(np.uint8)
                    
                        if (np.sum(object_masks[i]) > 8000):
                            big_mask_flag = True

                        # x, y, width, height
                        box = obj['bbox']
                        box_exp = []

                        offset = 2
                        new_x = box[0] - offset if box[0] >= offset else 0
                        box_exp.append(new_x)

                        new_y = box[1] - offset if box[1] >= offset else 0
                        box_exp.append(new_y)

                        new_w = box[2] + 2*offset if new_x + box[2] <= 127 else 127 - new_x
                        box_exp.append(new_w)

                        new_h = box[3] + 2*offset if new_y + box[3] <= 127 else 127 - new_y
                        box_exp.append(new_h)

                        obj_box = mask_tmp[box_exp[1] : box_exp[1] + box_exp[3], box_exp[0] : box_exp[0] + box_exp[2], :]

                        bg_pixel_coords = [0, 0]
                        bg_pixel_val = list(mask_tmp[bg_pixel_coords[1], bg_pixel_coords[0]])

                        tmp = np.logical_and((obj_box != obj_pixel_val).any(axis=2), (obj_box != bg_pixel_val).any(axis=2)).astype(np.int32)

                    if big_mask_flag:
                        print("TOO BIG MASK; SKIP")
                        continue

                    mask_tmp = mask.copy()
                    bg_pixel_coords = [0, 0]
                    bg_pixel_val = list(mask_tmp[bg_pixel_coords[1], bg_pixel_coords[0]])
                    
                    while (bg_pixel_val in obj_pixel_vals):
                        bg_pixel_coords[0] += 1
                        bg_pixel_val = list(mask_tmp[bg_pixel_coords[1], bg_pixel_coords[0]])

                    object_masks['bg'] = (mask_tmp == bg_pixel_val).all(axis=2).astype(np.uint8)


                    n_obj = len(scene_objs)
                    init_rel = np.array(['unlabelled' for _ in range(len(self.groups_rel))])
                    rel_index = {x : {y : init_rel.copy() for y in np.delete(np.arange(n_obj), x)} for x in np.arange(n_obj)}

                    for rel_name, obj_list in rels.items():
                        for i in self.groups_rel:
                            if rel_name in self.groups_rel[i]:
                                group_idx = i
                        for ref_idx, target_list in enumerate(obj_list):
                            for target_idx in target_list:
                                rel_index[ref_idx][target_idx][group_idx] = rel_name


                    for (ref_idx, target_list) in rel_index.items():
                        for (target_idx, rel_list) in target_list.items():
                            
                            scale = 1
                            color = cv2.resize(bgr.copy(), (0,0), fx=scale, fy=scale)
                            depth = cv2.resize(depth_orig.copy(), (0,0), fx=scale, fy=scale)
                            bg = cv2.resize(object_masks['bg'].copy(), (0,0), fx=scale, fy=scale)
                            ref = cv2.resize(object_masks[ref_idx].copy(), (0,0), fx=scale, fy=scale)
                            ref = ref.astype(np.float32)
                            tar = cv2.resize(object_masks[target_idx].copy(), (0,0), fx=scale, fy=scale)
                            tar = tar.astype(np.float32)

                            if np.sum(tar) == 0 or np.sum(ref) == 0 or (ref == tar).all():
                                continue

                            pixels = np.concatenate((color, depth[...,None], bg[...,None], ref[...,None], tar[...,None]), axis=2)

                            if array_name in train_files:
                                
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
                                    coords = scene_objs[idx]['3d_coords']
                                    if idx not in overlapping_objects:
                                        color = scene_objs[idx]['color']
                                        size = scene_objs[idx]['size']
                                        shape = scene_objs[idx]['shape']
                                        train_object_vectors[-1].append([object_colors.index(color),\
                                                                 object_shapes.index(shape),\
                                                                 object_sizes.index(size),\
                                                                 coords[0],\
                                                                 coords[1],\
                                                                 coords[2]])
                                        train_object_vector_masks[-1].append([1,1,1])
                                    else:
                                        color, size, shape = ['unlabelled', 'unlabelled', 'unlabelled']
                                        train_object_vectors[-1].append([0,\
                                                                 0,\
                                                                 0,\
                                                                 coords[0],\
                                                                 coords[1],\
                                                                 coords[2]])
                                        train_object_vector_masks[-1].append([0, 0, 0])

                            
                            elif array_name in test_files:
                                
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
                                    coords = scene_objs[idx]['3d_coords']
                                    if idx not in overlapping_objects:
                                        color = scene_objs[idx]['color']
                                        size = scene_objs[idx]['size']
                                        shape = scene_objs[idx]['shape']
                                        test_object_vectors[-1].append([object_colors.index(color),\
                                                                     object_shapes.index(shape),\
                                                                     object_sizes.index(size),\
                                                                     coords[0],\
                                                                     coords[1],\
                                                                     coords[2]])
                                        test_object_vector_masks[-1].append([1,1,1])
                                    else:
                                        color, size, shape = ['unlabelled', 'unlabelled', 'unlabelled']
                                        test_object_vectors[-1].append([0,\
                                                                 0,\
                                                                 0,\
                                                                 coords[0],\
                                                                 coords[1],\
                                                                 coords[2]])
                                        test_object_vector_masks[-1].append([0, 0, 0])

            if "unlabelled" not in ignore:
                array_list = sorted(os.listdir(folder_name_train_arr))[:]
                for array_name in array_list:

                    data = np.load(osp.join(folder_name_train_arr, array_name))['arr_0']
                    scene_file = open(osp.join(folder_name_train_scenes, array_name.replace('.npz', '.json')), "r")
                    json_data = json.load(scene_file)
                    scene_objs = json_data['objects']
                    rels = json_data['relationships']

                    bgr = (data[...,:3] * 255).astype(np.uint8)
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
                    bgr = bgr / 255.

                    depth_orig = data[...,3]
                    depth_orig = depth_orig / np.max(depth_orig)
                    
                    mask = data[...,4:]
                    mask = (mask * 255).astype(np.uint8)

                    object_masks = {}
                    obj_pixel_vals = []

                    bad_mask_flag = False

                    for i, obj in enumerate(scene_objs):
                        if 'mask_color' in obj:
                            obj_pixel_val = obj['mask_color']
                        else:
                            mask_tmp = mask.copy()
                            pixel_coords = obj['pixel_coords'][:2]
                            pixel_coords = [127 if x > 127 else x for x in pixel_coords]
                            pixel_coords = [0 if x < 0 else x for x in pixel_coords]

                            obj_pixel_val = mask_tmp[pixel_coords[1], pixel_coords[0]]
                        obj_pixel_vals.append(list(obj_pixel_val))

                        object_masks[i] = (mask_tmp == obj_pixel_val).all(axis=2).astype(np.uint8)
                    
                        if (np.sum(object_masks[i]) > 8000):
                            bad_mask_flag = True

                    if bad_mask_flag:
                        print("BAD MASKING; SKIP")
                        continue

                    mask_tmp = mask.copy()
                    bg_pixel_coords = [0, 0]
                    bg_pixel_val = list(mask_tmp[bg_pixel_coords[1], bg_pixel_coords[0]])
                    
                    while (bg_pixel_val in obj_pixel_vals):
                        bg_pixel_coords[0] += 1
                        bg_pixel_val = list(mask_tmp[bg_pixel_coords[1], bg_pixel_coords[0]])

                    object_masks['bg'] = (mask_tmp == bg_pixel_val).all(axis=2).astype(np.uint8)


                    n_obj = len(scene_objs)
                    init_rel = np.array(['unlabelled' for _ in range(len(self.groups_rel))])
                    rel_index = {x : {y : init_rel.copy() for y in np.delete(np.arange(n_obj), x)} for x in np.arange(n_obj)}

                    for rel_name, obj_list in rels.items():
                        for i in self.groups_rel:
                            if rel_name in self.groups_rel[i]:
                                group_idx = i
                        for ref_idx, target_list in enumerate(obj_list):
                            for target_idx in target_list:
                                rel_index[ref_idx][target_idx][group_idx] = rel_name


                    for (ref_idx, target_list) in rel_index.items():
                        for (target_idx, rel_list) in target_list.items():
                            
                            scale = 1
                            color = cv2.resize(bgr.copy(), (0,0), fx=scale, fy=scale)
                            depth = cv2.resize(depth_orig.copy(), (0,0), fx=scale, fy=scale)
                            bg = cv2.resize(object_masks['bg'].copy(), (0,0), fx=scale, fy=scale)
                            ref = cv2.resize(object_masks[ref_idx].copy(), (0,0), fx=scale, fy=scale)
                            ref = ref.astype(np.float32)
                            tar = cv2.resize(object_masks[target_idx].copy(), (0,0), fx=scale, fy=scale)
                            tar = tar.astype(np.float32)

                            pixels = np.concatenate((color, depth[...,None], bg[...,None], ref[...,None], tar[...,None]), axis=2)
                                
                            unseen[-1] += [pixels]
                            unseen_labels[-1].append(rel_list)
                            mask = []
                            vector = []

                            for i, rel_name in enumerate(rel_list):

                                if rel_name != "unlabelled":
                                    vector.append(self.groups_rel[i].index(rel_name))
                                    mask.append(1)
                                else:
                                    vector.append(0)
                                    mask.append(0)

                            unseen_masks[-1].append(mask)
                            unseen_vectors[-1].append(vector)

                            unseen_object_vectors[-1].append([])
                            unseen_object_vector_masks[-1].append([])
                            for idx in [ref_idx, target_idx]:
                                color = scene_objs[idx]['color']
                                size = scene_objs[idx]['size']
                                shape = scene_objs[idx]['shape']
                                coords = scene_objs[idx]['3d_coords']
                                unseen_object_vectors[-1][-1].append([object_colors.index(color),\
                                                                 object_shapes.index(shape),\
                                                                 object_sizes.index(size),\
                                                                 coords[0],\
                                                                 coords[1],\
                                                                 coords[2]])
                                unseen_object_vector_masks[-1][-1].append([1,1,1])

        train = np.array(train, dtype=np.float32)
        train_labels = np.array(train_labels)
        train_vectors = np.array(train_vectors)
        train_masks = np.array(train_masks)
        train_object_vectors = np.array(train_object_vectors)
        train_object_vector_masks = np.array(train_object_vector_masks)

        test = np.array(test, dtype=np.float32)
        test_labels = np.array(test_labels)
        test_vectors = np.array(test_vectors)
        test_masks = np.array(test_masks)
        test_object_vectors = np.array(test_object_vectors)
        test_object_vector_masks = np.array(test_object_vector_masks)
        
        unseen = np.array(unseen, dtype=np.float32)
        unseen_labels = np.array(unseen_labels)
        unseen_vectors = np.array(unseen_vectors)
        unseen_masks = np.array(unseen_masks)
        unseen_object_vectors = np.array(unseen_object_vectors)
        unseen_object_vector_masks = np.array(unseen_object_vector_masks)

        train = train.reshape([len(train)] + data_dimensions)
        test = test.reshape([len(test)] + data_dimensions)
        train = np.swapaxes(train, 1 ,3)
        test = np.swapaxes(test, 1 ,3)
        if unseen != []:
            unseen = np.swapaxes(unseen, 2 ,4)

        train_concat = TupleDataset(train, train_vectors, train_masks, \
                                    train_object_vectors, train_object_vector_masks)
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

        return result


if __name__ == "__main__":
    folder_names = ['photoreal_data/clevr_data_128_4_obj_' + str(i) for i in range(20)]
    data_generator = DataGenerator(folder_names=folder_names)
    result = data_generator.generate_dataset()