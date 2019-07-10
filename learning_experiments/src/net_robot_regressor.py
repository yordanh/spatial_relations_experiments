#!/usr/bin/env python
"""
title           :net_robot_regressor.py
description     :Contains definitions of all the models used in the experiments.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :11/2018
python_version  :2.7.14
==============================================================================
"""

import six

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L
from chainer.backends.cuda import get_device_from_array
import cupy as cp
from chainer import serializers

import numpy as np
import os.path as osp

import net_128x128_mvae as net_mvae
import data_generator_robot_data as data_generator

class Robot_Regressor(chainer.Chain):
    """Convolutional Variational AutoEncoder"""

    def __init__(self, in_channels_n=11, out_channels_n=6, alpha=1, beta=1, result_dir=""):
        super(Robot_Regressor, self).__init__()
        with self.init_scope():

            self.alpha = float(alpha)
            self.beta = float(beta)

            self.in_channels_n = in_channels_n
            self.out_channels_n = out_channels_n

            self.dense_channels = [256, 64, 64]


            #############################
            # Initialise the MVAE model #
            #############################
            if result_dir == "":
                RESULT_DIR = "multi_result_robot_exp"
            else:
                RESULT_DIR = result_dir
            # RESULT_DIR = "multi_result_robot_exp"
            
            # folder_names = ['yordan_experiments/off-on', 'yordan_experiments/nonfacing-facing', 'yordan_experiments/out-in']
            # folder_names = ['yordan_experiments/off-on']
            
            # generator = data_generator.DataGenerator(folder_names=folder_names)
            # train, train_labels, train_concat, _, test, test_labels, test_concat, _,\
            # _, _, _, _, groups_obj, groups_rel = generator.generate_dataset()

            # print("Train Shape: ", train.shape)
            # print("Test Shape: ", test.shape)

            possible_groups = [['off', 'on'], 
                               ['nonfacing', 'facing'],
                               ['out', 'in']]

            object_colors = ['red', 'blue', 'yellow', 'purple']
            object_shapes = ['cube', 'cup', 'bowl']

            # groups_rel = {0 : possible_groups[0]}
            #                1 : possible_groups[1],
            #                2 : possible_groups[2]}
            groups_rel = {i : possible_groups[i] for i in range(len(possible_groups))}

            groups_obj = {0 : object_colors,
                               1 : object_shapes}

            self.mvae = net_mvae.Conv_MVAE(3, latent_n=8, groups_obj=groups_obj, groups_rel=groups_rel, 
                                  alpha=1, beta=1, gamma_obj=1, gamma_rel=1, objects_n=2)
            
            for x in self.mvae.encoder:
                x.disable_update()

            for x in self.mvae.decoder:
                x.disable_update()

            for x in self.mvae.classifiers_obj:
                x.disable_update()

            for x in self.mvae.classifiers_rel:
                x.disable_update()

            for x in self.mvae.operators:
                x.disable_update()

            serializers.load_npz(osp.join(RESULT_DIR, str(0), "final.model"), self.mvae)
            chainer.cuda.get_device_from_id(0).use()
            self.mvae.to_gpu()


            #############################
            # Fully-connected regressor #
            #############################

            self.hidden_0 = L.Linear(self.in_channels_n, self.dense_channels[0])
            self.hidden_1 = L.Linear(self.dense_channels[0], self.dense_channels[1])
            self.hidden_2 = L.Linear(self.dense_channels[1], self.dense_channels[2])
            self.encoder_mu = L.Linear(self.dense_channels[2], self.out_channels_n)
            self.encoder_ln_var = L.Linear(self.dense_channels[2], self.out_channels_n)


    def __call__(self, x):

        latent, mu, ln_var = self.encode(x)
        
        # return latent
        return mu
        

    def encode(self, x):

        # feed images through the mvae model
        ref_obj_embedding, mu_obj, _ = self.mvae.get_latent_indiv(x)
        # ref_obj_embedding = mu_obj

        ref_obj_embedding = ref_obj_embedding[0]
        rel_embedding, rel_mu, _ = self.mvae.get_latent_rel(x)
        # rel_embedding = rel_mu

        embeddings = F.concat((ref_obj_embedding, rel_embedding), axis=1)

        dense_0_encoded = F.dropout(F.relu(self.hidden_0(embeddings)), ratio=0) # (1, 256)
        dense_1_encoded = F.dropout(F.relu(self.hidden_1(dense_0_encoded)), ratio=0) # (1, 256)
        # dense_2_encoded = F.dropout(F.relu(self.hidden_2(dense_1_encoded)), ratio=0)

        mu = self.encoder_mu(dense_1_encoded) # (1, 14)
        ln_var = self.encoder_ln_var(dense_1_encoded)  # (1, 14) log(sigma**2)

        return F.gaussian(mu, ln_var), mu, ln_var


    def lf_regress(self, in_img, in_rel_labels, rel_masks, object_labels, object_label_masks, eef, k=1):

        batchsize = float(len(in_img))
        
        x_true = eef.astype(cp.float32) # eef
        mvae_in = in_img # images, masks, labels

        mse_loss = 0
        kl = 0

        x_pred, mus, ln_vars = self.encode(mvae_in)
        x_pred = mus

        # KL TERM
        if self.beta != 0:
            kl += gaussian_kl_divergence(mus, ln_vars) / batchsize
        else:
            kl = chainer.Variable(cp.zeros(1).astype(cp.float32))

        # MSE TERM
        if self.alpha != 0:
            mse_loss += F.sum(F.mean_squared_error(x_true, x_pred)) / batchsize

        else:
            mse_loss = chainer.Variable(cp.zeros(1).astype(cp.float32))
            

        self.mse_loss = self.alpha * mse_loss
        self.kl = self.beta * kl

        self.loss = chainer.Variable(cp.zeros(1).astype(cp.float32))
        
        if self.alpha:
            self.loss += self.mse_loss
        if self.beta:
            self.loss += self.kl


        chainer.report({'loss': self.loss}, self)
        chainer.report({'mse_l': self.mse_loss}, self)
        chainer.report({'kl': self.kl}, self)

        return self.loss