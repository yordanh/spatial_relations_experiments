#!/usr/bin/env python
"""
title           :train_mvae.py
description     :Contains the main trainign loop and test time evaluation of the model.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :10/2018
python_version  :2.7.16
==============================================================================
"""

# Misc
import argparse
import os
import os.path as osp
import cv2
import numpy as np
import cupy
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import subprocess
import shutil
import json

# Chaier
import chainer
from chainer import training
from chainer.training import extensions
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu
import chainer.functions as F
from chainer import serializers

# Sibling Modules
import net_128x128_mvae as net
import data_generator_mvae as data_generator


def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--output_dir', '-o', default='result_mvae/',
                        help='Directory to output the result')
    parser.add_argument('--epochs', '-e', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--dimz', '-z', default=8, type=int,
                        help='Dimention of encoded vector')
    parser.add_argument('--batchsize', '-batch', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--beta', '-b', default=1,
                        help='Beta coefficient for the KL loss')
    parser.add_argument('--gamma_obj', '-gO', default=1,
                        help='Gamma coefficient for the OBJECT classification loss')
    parser.add_argument('--gamma_rel', '-gR', default=1,
                        help='Gamma coefficient for the RELATIONAL classification loss')
    parser.add_argument('--alpha', '-a', default=1, 
                        help='Alpha coefficient for the reconstruction loss')
    parser.add_argument('--freq', '-f', default=1000, 
                    help='Frequency at which snapshots of the model are saved.')
    parser.add_argument('--augment_counter', type=int, default=0, 
                    help='Number ot times to augment the train data')
    parser.add_argument('--objects_n', default=2, type=int,
                        help='# of objects to be used')

    args = parser.parse_args()

    if not osp.isdir(osp.join(args.output_dir)):
        os.makedirs(args.output_dir)

    print('\n###############################################')
    print('# GPU: \t\t\t{}'.format(args.gpu))
    print('# dim z: \t\t{}'.format(args.dimz))
    print('# Minibatch-size: \t{}'.format(args.batchsize))
    print('# Epochs: \t\t{}'.format(args.epochs))
    print('# Beta: \t\t{}'.format(args.beta))
    print('# Gamma OBJ: \t\t{}'.format(args.gamma_obj))
    print('# Gamma REL: \t\t{}'.format(args.gamma_rel))
    print('# Frequency: \t\t{}'.format(args.freq))
    print('# Out Folder: \t\t{}'.format(args.output_dir))
    print('###############################################\n')

    stats = {'train_loss': [], 'train_rec_loss': [], 'train_kl': [],
             'train_label_obj_acc': [], 'train_label_obj_loss': [],
             'train_label_rel_acc': [], 'train_label_rel_loss': [],
             'valid_loss': [], 'valid_rec_loss': [], 'valid_kl': [],
             'valid_label_obj_acc': [], 'valid_label_obj_loss': [],
             'valid_label_rel_acc': [], 'valid_label_rel_loss': []}

    folder = 'photoreal_data'
    folder_names = [osp.join(folder, 'clevr_data_128_4_obj_'+str(i)) for i in range(20)]
    
    generator = data_generator.DataGenerator(augment_counter=args.augment_counter, \
                                             folder_names=folder_names,\
                                             data_split=0.8)

    train, train_labels, train_concat, train_vectors, test, test_labels, test_concat, test_vectors,\
    unseen, unseen_labels, unseen_concat, unseen_vectors,\
    groups_obj, groups_rel = generator.generate_dataset(args=args)

    data_dimensions = train.shape
    print('\n###############################################')
    print("DATA_LOADED")
    print("# Training Images: \t\t{0}".format(train.shape))
    print("# Testing Images: \t\t{0}".format(test.shape))
    print("# Unseen Images: \t\t{0}".format(unseen.shape))
    print("# Training Rel Labels: \t\t{0}".format(train_labels.shape))
    print("# Testing Rel Labels: \t\t{0}".format(test_labels.shape))
    print("# Unseen Rel Labels: \t\t{0}".format(unseen_labels.shape))
    print("# Training Rel Vectors: \t\t{0}".format(train_vectors.shape))
    print("# Testing Rel Vectors: \t\t{0}".format(test_vectors.shape))
    print('###############################################\n')
    
    if len(train_concat[1]) > 0:
        print("# Relation Label Stats:")
        for group_idx, group in groups_rel.items():
            print("# Group: \t\t{0} : {1}".format(group_idx, group))
            for label_idx, label in enumerate(group + ["unlabelled"]):
                print("#{0} Train: \t\t{1}".format(label,len(filter(lambda x:label == x[group_idx], train_labels))))
                print("#{0} Test: \t\t{1}".format(label,len(filter(lambda x:label == x[group_idx], test_labels))))
        print('###############################################\n')

    if len(train_concat[3]) > 0:
        print("# Object Label Stats:")
        train_object_vectors = np.array([train_concat[i][3][j] for i in range(len(train_concat)) for j in range(args.objects_n)])
        test_object_vectors = np.array([test_concat[i][3][j] for i in range(len(test_concat)) for j in range(args.objects_n)])

        train_object_vector_masks = np.array([train_concat[i][4][j] for i in range(len(train_concat)) for j in range(args.objects_n)])
        test_object_vector_masks = np.array([test_concat[i][4][j] for i in range(len(test_concat)) for j in range(args.objects_n)])
        for group_idx, group in groups_obj.items():
            print("# Group: \t\t{0} : {1}".format(group_idx, group))
            for label_idx, label in enumerate(group):
                print("#{0} Train: \t\t{1}".format(label,len(filter(lambda (x, y):label_idx == x[group_idx] and y[group_idx] != 0, zip(train_object_vectors, train_object_vector_masks)))))
                print("#{0} Test: \t\t{1}".format(label,len(filter(lambda (x, y):label_idx == x[group_idx] and y[group_idx] != 0, zip(test_object_vectors, test_object_vector_masks)))))
            for label_idx, label in enumerate(["unlabelled"]):
                print("#{0} Train: \t\t{1}".format(label,len(filter(lambda (x, y):label_idx == x[group_idx] and y[group_idx] == 0, zip(train_object_vectors, train_object_vector_masks)))))
                print("#{0} Test: \t\t{1}".format(label,len(filter(lambda (x, y):label_idx == x[group_idx] and y[group_idx] == 0, zip(test_object_vectors, test_object_vector_masks)))))
        print('###############################################\n')

    train_iter = chainer.iterators.SerialIterator(train_concat, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_concat, args.batchsize,
                                                 repeat=False, shuffle=False)

    
    model = net.Conv_MVAE(train.shape[1], latent_n = args.dimz,
                          groups_obj = groups_obj, groups_rel = groups_rel, 
                          alpha=args.alpha, beta = args.beta, 
                          gamma_obj = args.gamma_obj, gamma_rel = args.gamma_rel,
                          objects_n = args.objects_n)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0005))

    updater = training.StandardUpdater(train_iter, optimizer, 
                                       loss_func=model.lf,
                                       device=args.gpu)

    trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.output_dir)
    trainer.extend(extensions.Evaluator(test_iter, model, eval_func=model.lf, device=args.gpu), name="val", trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport([
                                           'epoch', \
                                           'main/rec_l', 'val/main/rec_l', \
                                           'val/main/kl', \
                                           'main/obj_a','val/main/obj_a', \
                                           'main/rel_a','val/main/rel_a', \
                                           'main/obj_l', \
                                           'val/main/obj_l', \
                                           'main/rel_l',\
                                           'val/main/rel_l']))
    trainer.extend(extensions.PlotReport(['main/rec_l', \
                                          'val/main/rec_l'], \
                                           x_key='epoch', file_name='rec_loss.png', marker=None))
    trainer.extend(extensions.PlotReport(['main/kl', \
                                          'val/main/kl'], \
                                           x_key='epoch', file_name='kl.png', marker=None))
    trainer.extend(extensions.PlotReport(['main/obj_a', \
                                          'val/main/obj_a'], \
                                           x_key='epoch', file_name='object_acc.png', marker=None))
    trainer.extend(extensions.PlotReport(['main/obj_l', \
                                          'val/main/obj_l'], \
                                           x_key='epoch', file_name='object_loss.png', marker=None))
    trainer.extend(extensions.PlotReport(['main/rel_a', \
                                          'val/main/rel_a'], \
                                           x_key='epoch', file_name='relation_acc.png', marker=None))
    trainer.extend(extensions.PlotReport(['main/rel_l', \
                                          'val/main/rel_l'], \
                                           x_key='epoch', file_name='relation_loss.png', marker=None))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.FailOnNonNumber())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.trainer'), trigger=(args.epochs, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, filename='snapshot_epoch_{.updater.epoch}.model'), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'final.model'), trigger=(args.epochs, 'epoch'))
    trainer.extend(extensions.ExponentialShift('alpha', 0.5, init=1e-3, target=1e-8), trigger=(args.epochs/2, 'epoch')) # For Adam

    trainer.run()


if __name__ == '__main__':
    main()
