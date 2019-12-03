#!/usr/bin/env python
"""
title           :net_128x128_mvae.py
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

import numpy as np


class Operator(chainer.Chain):
    """Fully-connected feedforwards operator"""

    def __init__(self, input_channels=None, n_hidden=None, embed_size=None, group_n=None):
        super(Operator, self).__init__()
        with self.init_scope():

            self.n_hidden = n_hidden
            self.embed_size = embed_size
            
            self.operator_0 = L.Linear(input_channels, self.n_hidden)

            self.operator_mu = L.Linear(self.n_hidden, self.embed_size)
            self.operator_ln_var = L.Linear(self.n_hidden, self.embed_size)

    def __call__(self, z_concat):
        z = self.encode(z_concat)
        return z

    def encode(self, z_concat):

        z_tmp0 = F.dropout(F.relu(self.operator_0(z_concat)), ratio=0)

        mu = self.operator_mu(z_tmp0)
        ln_var = self.operator_ln_var(z_tmp0)
        
        return mu, ln_var

class Conv_MVAE(chainer.Chain):
    """Convolutional Variational AutoEncoder"""

    def __init__(self, in_channels_n=None, latent_n=None, groups_obj=None, groups_rel=None, \
                 alpha=1, beta=1, gamma_obj=1, gamma_rel=1, objects_n=2):
        super(Conv_MVAE, self).__init__()
        with self.init_scope():

            self.alpha = float(alpha)
            self.beta = float(beta)
            self.gamma_obj = float(gamma_obj)
            self.gamma_rel = float(gamma_rel)

            self.in_channels_n = in_channels_n
            self.latent_n = latent_n
            
            self.groups_obj_len = [len(groups_obj[key]) for key in sorted(groups_obj.keys())]
            self.groups_obj_n = len(groups_obj.keys())
            self.groups_rel_len = [len(groups_rel[key]) for key in sorted(groups_rel.keys())]
            self.groups_rel_n = len(groups_rel.keys())
            
            self.rgb_channels_n = 3
            self.bg_channel_n = 1
            self.depth_channel_n = 1
            self.objects_n = int(objects_n)
            
            self.operators = chainer.ChainList()
            self.classifiers_rel = chainer.ChainList()
            self.classifiers_obj = chainer.ChainList()

            self.conv_channels = [32,32,64,64,64]
            self.dense_channels = [1024,256]

            self.encoder_conv_0 = L.Convolution2D(5, self.conv_channels[0], ksize=3, pad=1, stride=2) # (64, 64)
            self.encoder_conv_1 = L.Convolution2D(self.conv_channels[0], self.conv_channels[1], ksize=3, pad=1, stride=2) # (32, 32)
            self.encoder_conv_2 = L.Convolution2D(self.conv_channels[1], self.conv_channels[2], ksize=3, pad=1, stride=2) # (16, 16)
            self.encoder_conv_3 = L.Convolution2D(self.conv_channels[2], self.conv_channels[3], ksize=3, pad=1, stride=2) # (8, 8)
            self.encoder_conv_4 = L.Convolution2D(self.conv_channels[3], self.conv_channels[4], ksize=3, pad=1, stride=2) # (4, 4)
            # reshape from (64, 4, 4) to (1, 1024)
            self.encoder_dense_0 = L.Linear(self.dense_channels[0], self.dense_channels[1])

            self.encoder_mu = L.Linear(self.dense_channels[1], self.latent_n)
            self.encoder_ln_var = L.Linear(self.dense_channels[1], self.latent_n)

            for i in range(self.groups_obj_n):
                self.classifiers_obj.add_link(L.Linear(1, self.groups_obj_len[i]))

            for i in range(self.groups_rel_n):
                self.classifiers_rel.add_link(L.Linear(1, self.groups_rel_len[i]))

            for i in range(self.groups_rel_n):
                self.operators.add_link(Operator(input_channels=2*(self.latent_n), 
                                                 n_hidden=256, 
                                                 embed_size=1,
                                                 group_n=self.groups_rel_n))


            #####################
            ## spatial decoder ##
            #####################

            self.sp_dec_3 = L.Convolution2D(self.latent_n+2, 64, ksize=3, pad=1)
            self.sp_dec_2 = L.Convolution2D(64, 64, ksize=3, pad=1)
            self.sp_dec_1 = L.Convolution2D(64, 64, ksize=3, pad=1)
            self.sp_dec_0 = L.Convolution2D(64, 5, ksize=3, pad=1)


            # Add these to be able to disable them later
            self.encoder = [self.encoder_conv_0,
                            self.encoder_conv_1,
                            self.encoder_conv_2,
                            self.encoder_conv_3,
                            self.encoder_conv_4,
                            self.encoder_dense_0,
                            self.encoder_mu,
                            self.encoder_ln_var]

            self.decoder = [self.sp_dec_3,
                            self.sp_dec_2,
                            self.sp_dec_1,
                            self.sp_dec_0]

    def __call__(self, x):

        latents = []
        in_img = x
        shape = (in_img.shape[0], 5, in_img.shape[2], in_img.shape[3])
        out_img_overall = chainer.Variable(cp.zeros(shape).astype(cp.float32))
        out_img_indiv = []

        offset_channels_n = self.rgb_channels_n + self.bg_channel_n + self.depth_channel_n
        for obj_idx in range(offset_channels_n, self.objects_n + offset_channels_n):
            rec_mask = in_img[:,obj_idx][:, None, :, :]

            latent, mu, ln_var = self.encode(F.concat((in_img[:, :4], rec_mask), axis=1))

            latents.append(latent)

            out_img_infer = self.spatial_decode(latent)
            out_img_indiv.append(out_img_infer)

            if obj_idx != 0:
                out_img_overall += out_img_infer * rec_mask
        
        return out_img_overall, out_img_indiv
        
    def encode(self, x):
        conv_0_encoded = F.relu(self.encoder_conv_0(x))
        conv_1_encoded = F.relu(self.encoder_conv_1(conv_0_encoded))
        conv_2_encoded = F.relu(self.encoder_conv_2(conv_1_encoded))
        conv_3_encoded = F.relu(self.encoder_conv_3(conv_2_encoded))
        conv_4_encoded = F.relu(self.encoder_conv_4(conv_3_encoded))

        reshaped_encoded = F.reshape(conv_4_encoded, (len(conv_4_encoded), 1, self.dense_channels[0]))
        dense_0_encoded = F.dropout(F.relu(self.encoder_dense_0(reshaped_encoded)), ratio=0)
        mu = self.encoder_mu(dense_0_encoded)
        ln_var = self.encoder_ln_var(dense_0_encoded)

        return F.gaussian(mu, ln_var), mu, ln_var
        

    def decode(self, latent, sigmoid=True):

        dense_0_decoded = self.decoder_dense_0(latent)
        dense_1_decoded = self.decoder_dense_1(dense_0_decoded)
        reshaped_decoded = F.reshape(dense_1_decoded, (len(dense_1_decoded), self.conv_channels[3], 8, 8))
        up_4_decoded = F.upsampling_2d(reshaped_decoded, cp.ones(reshaped_decoded.shape), ksize=2, cover_all=False)
        deconv_3_decoded = F.relu(self.decoder_conv_3(up_4_decoded))
        up_3_decoded = F.upsampling_2d(deconv_3_decoded, cp.ones(deconv_3_decoded.shape), ksize=2, cover_all=False)
        deconv_2_decoded = F.relu(self.decoder_conv_2(up_3_decoded))
        up_2_decoded = F.upsampling_2d(deconv_2_decoded, cp.ones(deconv_2_decoded.shape), ksize=2, cover_all=False)
        out_img = self.decoder_conv_1(up_2_decoded)
        
        # need the check because the bernoulli_nll has a sigmoid in it
        if sigmoid:
            return F.sigmoid(out_img)
        else:
            return out_img


    def spatial_decode(self, latent, sigmoid=True):

        image_size = 128

    	a = cp.linspace(-1, 1, image_size)
    	b = cp.linspace(-1, 1, image_size)

    	x, y = cp.meshgrid(a, b)

    	x = x.reshape(image_size, image_size, 1)
    	y = y.reshape(image_size, image_size, 1)

    	batchsize = len(latent)

    	xy = cp.concatenate((x,y), axis=-1)
    	xy_tiled = cp.tile(xy, (batchsize, 1, 1, 1)).astype(cp.float32)

    	latent_tiled = F.tile(latent, (1, 1, image_size*image_size)).reshape(batchsize, image_size, image_size, self.latent_n)
    	latent_and_xy = F.concat((latent_tiled, xy_tiled), axis=-1)
    	latent_and_xy = F.swapaxes(latent_and_xy, 1, 3)

    	sp_3_decoded = F.relu(self.sp_dec_3(latent_and_xy))
        sp_2_decoded = F.relu(self.sp_dec_2(sp_3_decoded))
    	sp_1_decoded = F.relu(self.sp_dec_1(sp_2_decoded))
    	out_img = self.sp_dec_0(sp_1_decoded)

    	# need the check because the bernoulli_nll has a sigmoid in it
        if sigmoid:
            return F.sigmoid(out_img)
        else:
            return out_img

    def predict_rel_label(self, latent_rel, softmax=True):
        result = []
        
        for i in range(self.groups_rel_n):
            prediction = self.classifiers_rel[i](latent_rel[:, i, None])

            # need the check because the softmax_cross_entropy has a softmax in it
            if softmax:
                result.append(F.softmax(prediction))
            else:
                result.append(prediction)

        return result


    def predict_obj_label(self, latent, softmax=True):
        result = []
        
        for i in range(self.groups_obj_n):
            prediction = self.classifiers_obj[i](latent[:, i, None])

            # need the check because the softmax_cross_entropy has a softmax in it
            if softmax:
                result.append(F.softmax(prediction))
            else:
                result.append(prediction)

        return result


    # predict the latent vector for each object
    def get_latent_indiv(self, x):
        latents = []
        mus = []
        ln_vars = []

        in_img = x

        offset_channels_n = self.rgb_channels_n + self.bg_channel_n + self.depth_channel_n
        for obj_idx in range(offset_channels_n, self.objects_n + offset_channels_n):
            rec_mask = in_img[:, obj_idx][:, None, :, :]

            latent, mu, ln_var = self.encode(F.concat((in_img[:, :4], rec_mask), axis=1))

            mus.append(mu)
            ln_vars.append(ln_var)

            latents.append(latent)
        
        return latents, mus, ln_vars


    # predict the latent vector for the relationship
    def get_latent_rel(self, x):
        latents = []
        mus = []
        ln_vars = []

        latents, mus_obj, _ = self.get_latent_indiv(x)

        latent_concat = F.concat((latents), axis=1)

        for i in range(self.groups_rel_n):
            
            mu, ln_var = self.operators[i](latent_concat)
            
            mus.append(mu)
            ln_vars.append(ln_var)

        mus = F.concat((mus), axis=1)
        ln_vars = F.concat((ln_vars), axis=1)

        latent_rel = F.gaussian(mus, ln_vars)

        return latent_rel, mus, ln_vars


    def check_loss_coefficients(self):
        @chainer.training.make_extension()
        def f(trainer):
            if trainer.updater.epoch >= 0:
                self.gamma_rel = 50000 


        return f


    def lf(self, in_img, in_rel_labels, rel_masks, object_labels, object_label_masks, k=1):

        batchsize = float(len(in_img))
        denom = (k * batchsize * self.objects_n)

        
        latents = []
        rec_loss = 0
        kl = 0
        label_obj_loss = 0
        label_obj_acc = 0
        label_rel_loss = 0
        label_rel_acc = 0

        latents, mus, ln_vars = self.get_latent_indiv(in_img)

        offset_channels_n = self.rgb_channels_n + self.bg_channel_n + self.depth_channel_n

        for obj_idx in range(self.objects_n):

            rec_mask = in_img[:,obj_idx + offset_channels_n][:, None, :, :]

            # KL TERM
            if self.beta != 0:
                kl += gaussian_kl_divergence(mus[obj_idx], ln_vars[obj_idx]) / denom
            else:
                kl = chainer.Variable(cp.zeros(1).astype(cp.float32))


            # RESAMPLING
            for l in six.moves.range(k):          

                # RECONSTRUCTION TERM
                if self.alpha != 0:
                    out_img = self.spatial_decode(latents[obj_idx], sigmoid=False)

                    x_true = in_img[:, :self.rgb_channels_n + self.depth_channel_n]
                    x_pred = out_img[:, :self.rgb_channels_n + self.depth_channel_n]
                    rec_loss += F.sum(F.bernoulli_nll(x_true, x_pred, reduce="no") * rec_mask) / denom

                    x_true = in_img[:, obj_idx + offset_channels_n]
                    x_pred = out_img[:, 4]
                    rec_loss += F.bernoulli_nll(x_true, x_pred) / denom

                else:
                    rec_loss = chainer.Variable(cp.zeros(1).astype(cp.float32))
            
                # OBJECT CLASSIFICATION TERM
                if self.gamma_obj != 0:

                    out_obj_labels = self.predict_obj_label(latents[obj_idx], softmax=False)
                    in_obj_labels = object_labels[:, obj_idx, :self.groups_obj_n].astype(cp.int32)
                    masks = object_label_masks[:, obj_idx].astype(cp.float32)

                    for i in range(self.groups_obj_n):
                        
                        o_mask = masks[:, i].astype(cp.float32)

                        if F.sum(o_mask).data == 0:
                            label_obj_loss += 0
                            label_obj_acc += 1 / (k * self.objects_n)
                            continue

                        label_obj_loss += F.sum(F.softmax_cross_entropy(out_obj_labels[i], in_obj_labels[:, i], reduce='no') * o_mask) / (k * F.sum(o_mask) * self.objects_n)
                        
                        in_aug_obj_labels = (in_obj_labels[:, i] * o_mask + (100*(1 - o_mask))).astype(cp.int32)
                        label_obj_acc += F.accuracy(out_obj_labels[i], in_aug_obj_labels, ignore_label=100) / (k * self.objects_n)
                else:
                    label_obj_loss = chainer.Variable(cp.zeros(1).astype(cp.float32))
                    label_obj_acc = chainer.Variable(cp.zeros(1).astype(cp.float32))


        #########################################
        ############# RELATIONAL LABELS #########
        #########################################

        latent_concat = F.concat((latents), axis=1)
        mus_rel = []
        ln_vars_rel = []

        for i in range(self.groups_rel_n):
            mu_rel, ln_var_rel = self.operators[i](latent_concat)
            mus_rel.append(mu_rel)
            ln_vars_rel.append(ln_var_rel)

        mus_rel = F.concat((mus_rel), axis=1)
        ln_vars_rel = F.concat((ln_vars_rel), axis=1)

        latent_rel = F.gaussian(mus_rel, ln_vars_rel)
        out_rel_labels = self.predict_rel_label(latent_rel, softmax=False)


        # KL TERM
        if self.beta != 0:
            kl += gaussian_kl_divergence(mus_rel, ln_vars_rel) / batchsize
        else:
            kl = chainer.Variable(cp.zeros(1).astype(cp.float32))

        if self.gamma_rel != 0:

            for i in range(self.groups_rel_n):
                r_mask = rel_masks[:, i].astype(cp.float32)
                if F.sum(r_mask).data == 0:
                    label_rel_loss += 0
                    label_rel_acc += 1
                    continue 

                label_rel_loss += F.sum(F.softmax_cross_entropy(out_rel_labels[i], in_rel_labels[:, i], reduce='no') * r_mask) / (k * F.sum(r_mask))
                
                in_aug_rel_labels = (in_rel_labels[:, i] * r_mask + (100*(1 - r_mask))).astype(cp.int32)
                label_rel_acc += F.accuracy(out_rel_labels[i], in_aug_rel_labels, ignore_label=100) / (k)
        else:
            label_rel_loss = chainer.Variable(cp.zeros(1).astype(cp.float32))
            label_rel_acc = chainer.Variable(cp.zeros(1).astype(cp.float32))

        #########################################
        ############# RELATIONAL LABELS #########
        #########################################
            
        self.total_corr = chainer.Variable(cp.zeros(1).astype(cp.float32))

        self.rec_loss = self.alpha * rec_loss
        self.kl = self.beta * kl
        self.label_obj_loss = self.gamma_obj * label_obj_loss
        self.label_obj_acc = label_obj_acc
        self.label_rel_loss = self.gamma_rel * label_rel_loss
        self.label_rel_acc = label_rel_acc

        self.loss = chainer.Variable(cp.zeros(1).astype(cp.float32))
        
        if self.alpha:
            self.loss += self.rec_loss
        if self.beta:
            self.loss += self.kl
        if self.gamma_obj:
            self.loss += self.label_obj_loss
        if self.gamma_rel:
            self.loss += self.label_rel_loss


        chainer.report({'loss': self.loss}, self)
        chainer.report({'rec_l': self.rec_loss}, self)
        chainer.report({'kl': self.kl}, self)
        chainer.report({'obj_l': self.label_obj_loss}, self)
        chainer.report({'obj_a': self.label_obj_acc}, self)
        chainer.report({'rel_l': self.label_rel_loss}, self)
        chainer.report({'rel_a': self.label_rel_acc}, self)
        
        return self.loss

