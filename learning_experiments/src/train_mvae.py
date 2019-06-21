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
from config_parser import ConfigParser
from utils import *


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

    if not osp.isdir(osp.join(args.output_dir, 'models')):
        os.makedirs(osp.join(args.output_dir, 'models'))

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

    models_folder = os.path.join(args.output_dir, "models")

    n_obj = 3
    folder = 'clevr_data_128_'+str(n_obj)+'_obj'
    folder_names = [osp.join(folder, folder+'_'+str(i)) for i in range(145, 150)]

    # n_obj = 3
    # folder = 'clevr_data_128_'+str(n_obj)+'_obj'
    # folder_names += [osp.join(folder, folder+'_'+str(i)) for i in range(60, 70)]
    
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
    # optimizer = chainer.optimizers.RMSprop()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0005))
    # optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(0.00001))

    lf = model.get_loss_func()


    # vs = model.get_loss_func()(cupy.asarray(test[:2]))
    # # vs = model(test)
    # import chainer.computational_graph as c
    # g = c.build_computational_graph(vs)
    # with open('./result/file_latent.dot', 'w') as o:
    # # with open('./result/file_rec.dot', 'w') as o:
    #     o.write(g.dump())
    # exit()

    
    stats, model, optimizer, _ = training_loop(model=model, optimizer=optimizer, stats=stats, 
                                                           epochs=args.epochs, train_iter=train_iter, 
                                                           test_iter=test_iter, lf=lf, 
                                                           models_folder=models_folder, 
                                                           mode="supervised", args=args)

    print("Save Stats\n")
    np.savez(os.path.join(args.output_dir, 'stats.npz'), **stats)

    print("Save Model\n")
    serializers.save_npz(os.path.join(models_folder, 'final.model'), model)

    print("Save Optimizer\n")
    serializers.save_npz(os.path.join(models_folder, 'final.state'), optimizer)


def training_loop(model=None, optimizer=None, stats=None, epochs=None, train_iter=None, test_iter=None, 
                  lf=None, models_folder=None, epochs_so_far=0, mode=None, args=None):

    train_losses = []
    train_rec_losses = []
    train_kl = []
    train_label_obj_losses = []
    train_label_obj_accs = []
    train_label_rel_losses = []
    train_label_rel_accs = []

    train_dist = []
    train_total_corr = []

    while train_iter.epoch < epochs:
        # ------------ One epoch of the training loop ------------
        # ---------- One iteration of the training loop ----------

        # if train_iter.epoch < 20:
        #     model.gamma = 0
        #     # model.alpha = 0
        # else:
        #   model.gamma = float(args.gamma)
        #     # model.alpha = float(args.alpha)

        # model.beta = float(args.beta) * (train_iter.epoch / float(100))

        train_batch = train_iter.next()

        image_train = concat_examples(train_batch, 0)

        # Calculate the loss with softmax_cross_entropy
        loss, rec_loss, kl,label_obj_loss, label_obj_acc,\
        label_rel_loss, label_rel_acc = model.get_loss_func()(image_train)

        train_losses.append(loss.array)
        train_rec_losses.append(rec_loss.array)
        train_kl.append(kl.array)
        train_label_obj_losses.append(label_obj_loss.array)
        train_label_obj_accs.append(label_obj_acc.array)
        train_label_rel_losses.append(label_rel_loss.array)
        train_label_rel_accs.append(label_rel_acc.array)

        model.cleargrads()
        loss.backward()

        # Update all the trainable paremters
        optimizer.update()


        if train_iter.epoch % int(args.freq) == 0:
            serializers.save_npz(os.path.join(models_folder ,str(train_iter.epoch + epochs_so_far) + '.model'), model)

        # --------------------- iteration until here --------------------- 

        if train_iter.is_new_epoch:

            valid_losses = []
            valid_rec_losses = []
            valid_kl = []
            valid_label_obj_losses = []
            valid_label_obj_accs = []
            valid_label_rel_losses = []
            valid_label_rel_accs = []

            while True:

                test_batch = test_iter.next()

                image_test = concat_examples(test_batch, 0)

                loss, rec_loss, kl,label_obj_loss, label_obj_acc,\
                label_rel_loss, label_rel_acc = model.get_loss_func()(image_test)

                valid_losses.append(loss.array)
                valid_rec_losses.append(rec_loss.array)
                valid_kl.append(kl.array)
                valid_label_obj_losses.append(label_obj_loss.array)
                valid_label_obj_accs.append(label_obj_acc.array)
                valid_label_rel_losses.append(label_rel_loss.array)
                valid_label_rel_accs.append(label_rel_acc.array)

                if test_iter.is_new_epoch:
                    test_iter.epoch = 0
                    test_iter.current_position = 0
                    test_iter.is_new_epoch = False
                    test_iter._pushed_position = None
                    break

            stats['train_loss'].append(np.mean(to_cpu(train_losses)))
            stats['train_rec_loss'].append(np.mean(to_cpu(train_rec_losses)))
            stats['train_label_obj_loss'].append(np.mean(to_cpu(train_label_obj_losses)))
            stats['train_label_obj_acc'].append(np.mean(to_cpu(train_label_obj_accs)))
            stats['train_label_rel_loss'].append(np.mean(to_cpu(train_label_rel_losses)))
            stats['train_label_rel_acc'].append(np.mean(to_cpu(train_label_rel_accs)))

            stats['valid_loss'].append(np.mean(to_cpu(valid_losses)))
            stats['valid_rec_loss'].append(np.mean(to_cpu(valid_rec_losses)))
            stats['valid_label_obj_loss'].append(np.mean(to_cpu(valid_label_obj_losses)))
            stats['valid_label_obj_acc'].append(np.mean(to_cpu(valid_label_obj_accs)))
            stats['valid_label_rel_loss'].append(np.mean(to_cpu(valid_label_rel_losses)))
            stats['valid_label_rel_acc'].append(np.mean(to_cpu(valid_label_rel_accs)))

            stats['valid_kl'].append(np.mean(to_cpu(valid_kl)))
            stats['train_kl'].append(np.mean(to_cpu(train_kl)))
    
            # print(("Ep: {0}\tT: {1}\tV: {2}\tT_R: {3}\tV_R: {4}\t" + \
            #       "T_KL: {5}\tV_KL: {6}\tT_Acc: {7}\tV_Acc: {8}\t" + \
            #       "T_LL: {9}\tV_LL: {10}").format(train_iter.epoch, 
            #                                                     round(stats['train_loss'][-1], 2),
            #                                                     round(stats['valid_loss'][-1], 2),
            #                                                     round(stats['train_rec_loss'][-1], 2),
            #                                                     round(stats['valid_rec_loss'][-1], 2),
            #                                                     round(stats['train_kl'][-1], 2),
            #                                                     round(stats['valid_kl'][-1], 2),
            #                                                     round(stats['train_label_acc'][-1], 4),
            #                                                     round(stats['valid_label_acc'][-1], 4),
            #                                                     round(stats['train_label_loss'][-1], 2),
            #                                                     round(stats['valid_label_loss'][-1], 2)))

            print(("Ep: {0}\t" + \
                  "T_R: {1}\t" +\
                  "V_R: {2}\t" + \
                  "T_KL: {3}\t" +\
                  "V_KL: {4}\t" + \
                  "T_O_Acc: {5}\t" +\
                  "V_O_Acc: {6}\t" + \
                  "T_R_Acc: {7}\t" +\
                  "V_R_Acc: {8}\t" + \
                  # "T_O_LL: {9}\t" +\
                  "V_O_LL: {10}\t" + \
                  # "T_R_LL: {11}\t" +\
                  "V_R_LL: {12}").format(train_iter.epoch, 
                                                    round(stats['train_rec_loss'][-1], 0),
                                                    round(stats['valid_rec_loss'][-1], 0),
                                                    round(stats['train_kl'][-1], 0),
                                                    round(stats['valid_kl'][-1], 0),
                                                    round(stats['train_label_obj_acc'][-1], 4),
                                                    round(stats['valid_label_obj_acc'][-1], 4),
                                                    round(stats['train_label_rel_acc'][-1], 4),
                                                    round(stats['valid_label_rel_acc'][-1], 4),
                                                    round(stats['train_label_obj_loss'][-1], 0),
                                                    round(stats['valid_label_obj_loss'][-1], 0),
                                                    round(stats['train_label_rel_loss'][-1], 0),
                                                    round(stats['valid_label_rel_loss'][-1], 0)))

            train_losses = []
            train_rec_losses = []
            train_kl = []
            train_label_obj_losses = []
            train_label_obj_accs = []
            train_label_rel_losses = []
            train_label_rel_accs = []

    return stats, model, optimizer, epochs


if __name__ == '__main__':
    main()
