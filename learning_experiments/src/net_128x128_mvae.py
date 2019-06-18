#!/usr/bin/env python
"""
title           :net_200x200_mvae.py
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
            # self.operator_1 = L.Linear(self.n_hidden, self.n_hidden / 2)
            # self.operator_mu = L.Linear(self.n_hidden / 2, self.embed_size)
            # self.operator_ln_var = L.Linear(self.n_hidden / 2, self.embed_size)

            # self.deoperator_z = L.Linear(group_n + (input_channels / 2), self.n_hidden)
            # self.deoperator_1 = L.Linear(self.n_hidden, self.n_hidden)
            # self.deoperator_0 = L.Linear(self.n_hidden, 3)



            self.operator_mu = L.Linear(self.n_hidden, self.embed_size)
            self.operator_ln_var = L.Linear(self.n_hidden, self.embed_size)

            self.deoperator_0 = L.Linear(group_n + input_channels / 2, 3)


    def __call__(self, z_concat):
        """AutoEncoder"""
        z = self.encode(z_concat)
        return z

    def encode(self, z_concat):

        z_tmp0 = F.relu(self.operator_0(z_concat))
        # z_tmp1 = F.dropout(F.relu(self.operator_1(z_tmp0)), ratio=0.5)
        
        # mu = self.operator_mu(z_tmp1)
        # ln_var = self.operator_ln_var(z_tmp1)

        mu = self.operator_mu(z_tmp0)
        ln_var = self.operator_ln_var(z_tmp0)
        
        return mu, ln_var


    def decode(self, latent_z):

        # tmp_0 = F.relu(self.deoperator_z(latent_z))
        # tmp_1 = F.relu(self.deoperator_1(tmp_0))
        # zs = self.deoperator_0(tmp_1)

        zs = self.deoperator_0(latent_z)

        return zs


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

            self.conv_channels = [32,32,32,64,64,64]
            self.dense_channels = [1024,256]

            ########################
            # encoder for branch 0 #
            ######################## 
            self.encoder_conv_0 = L.Convolution2D(5, self.conv_channels[0], ksize=3, pad=1, stride=2) # (64, 64)
            self.encoder_conv_1 = L.Convolution2D(self.conv_channels[0], self.conv_channels[1], ksize=3, pad=1, stride=2) # (32, 32)
            self.encoder_conv_2 = L.Convolution2D(self.conv_channels[1], self.conv_channels[2], ksize=3, pad=1, stride=2) # (16, 16)
            self.encoder_conv_3 = L.Convolution2D(self.conv_channels[2], self.conv_channels[3], ksize=3, pad=1, stride=2) # (8, 8)
            self.encoder_conv_4 = L.Convolution2D(self.conv_channels[3], self.conv_channels[4], ksize=3, pad=1, stride=2) # (4, 4)
            self.encoder_conv_5 = L.Convolution2D(self.conv_channels[4], self.conv_channels[5], ksize=3, pad=1, stride=2) # (2, 2)
            # reshape from (8, 8, 8) to (1,512)
            self.encoder_dense_0 = L.Linear(self.dense_channels[0], self.dense_channels[1])
            # self.encoder_dense_1 = L.Linear(self.dense_channels[6], self.dense_channels[7])

            self.encoder_mu = L.Linear(self.dense_channels[1], self.latent_n)
            self.encoder_ln_var = L.Linear(self.dense_channels[1], self.latent_n)

            # self.operators = chainer.ChainList()
            # for _ in range(3):
            #     self.operators.add_link(L.Linear(self.latent_n + 3, 2))

            for i in range(self.groups_obj_n):
                self.classifiers_obj.add_link(L.Linear(1, self.groups_obj_len[i]))

            for i in range(self.groups_rel_n):
                self.classifiers_rel.add_link(L.Linear(1, self.groups_rel_len[i]))

            for i in range(self.groups_rel_n):
                self.operators.add_link(Operator(input_channels=2*(self.latent_n), 
                                                 n_hidden=256, 
                                                 embed_size=1,
                                                 group_n=self.groups_rel_n))


            ########################
            # decoder for branch 0 #
            ########################
            self.decoder_dense_0 = L.Linear(self.latent_n, self.dense_channels[1])
            self.decoder_dense_1 = L.Linear(self.dense_channels[1], self.dense_channels[0])
            # self.decoder_dense_2 = L.Linear(self.dense_channels[6], self.dense_channels[5])
            # reshape from (1, 512) to (8, 8, 8)
            self.decoder_conv_5 = L.Convolution2D(self.conv_channels[5], self.conv_channels[4], ksize=3, pad=1) # (4, 4)
            self.decoder_conv_4 = L.Convolution2D(self.conv_channels[4], self.conv_channels[3], ksize=3, pad=1) # (8, 8)
            self.decoder_conv_3 = L.Convolution2D(self.conv_channels[3], self.conv_channels[2], ksize=3, pad=1) # (16, 16)
            self.decoder_conv_2 = L.Convolution2D(self.conv_channels[2], self.conv_channels[1], ksize=3, pad=1) # (32, 32)
            self.decoder_conv_1 = L.Convolution2D(self.conv_channels[1], 5, ksize=3, pad=1) # (64, 64)
            self.decoder_conv_0 = L.Convolution2D(self.conv_channels[0], 5, ksize=3, pad=1) # (128, 128)



            #####################
            ## spatial decoder ##
            #####################

            self.sp_dec_3 = L.Convolution2D(self.latent_n+2, 64, ksize=3, pad=1)
            self.sp_dec_2 = L.Convolution2D(64, 64, ksize=3, pad=1)
            self.sp_dec_1 = L.Convolution2D(64, 64, ksize=3, pad=1)
            self.sp_dec_0 = L.Convolution2D(64, 5, ksize=3, pad=1)


    def __call__(self, x):

        latents = []
        in_img = x
        shape = (in_img.shape[0], 5, in_img.shape[2], in_img.shape[3])
        out_img_overall = chainer.Variable(cp.zeros(shape).astype(cp.float32))
        out_img_indiv = []

        offset_channels_n = self.rgb_channels_n + self.bg_channel_n + self.depth_channel_n
        for obj_idx in range(offset_channels_n, self.objects_n + offset_channels_n):
            rec_mask = in_img[:,obj_idx][:, None, :, :]

            rec_mask[rec_mask == 0.25] = 0
            rec_mask[rec_mask == 0.75] = 1

            # latent, mu, ln_var = self.encode(in_img[:, :3])
            latent, mu, ln_var = self.encode(F.concat((in_img[:, :4], rec_mask), axis=1))
            # latent, mu, ln_var = self.encode(rec_mask)

            latents.append(latent)

            out_img_infer = self.spatial_decode(latent)
            # out_img_infer = F.concat((in_img[:, :3], rec_mask), axis=1) * rec_mask
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
        # conv_5_encoded = F.relu(self.encoder_conv_5(conv_4_encoded))

        reshaped_encoded = F.reshape(conv_4_encoded, (len(conv_4_encoded), 1, self.dense_channels[0])) # (1, 512)
        dense_0_encoded = F.dropout(F.relu(self.encoder_dense_0(reshaped_encoded)), ratio=0) # (1, 8)
        # dense_1_encoded = F.relu(self.encoder_dense_1(dense_0_encoded)) # (1, 8)
        mu = self.encoder_mu(dense_0_encoded) # (1, 2)
        ln_var = self.encoder_ln_var(dense_0_encoded)  # (1, 2) log(sigma**2)

        return F.gaussian(mu, ln_var), mu, ln_var
        

    def decode(self, latent, sigmoid=True):

        dense_0_decoded = self.decoder_dense_0(latent) # (1, 10)
        dense_1_decoded = self.decoder_dense_1(dense_0_decoded) # (1, 512)
        # dense_2_decoded = F.relu(self.decoder_dense_2(dense_1_decoded)) # (1, 512)
        reshaped_decoded = F.reshape(dense_1_decoded, (len(dense_1_decoded), self.conv_channels[3], 8, 8))# (8, 8)
        # deconv_5_decoded = F.relu(self.decoder_conv_5(latent))
        # up_5_decoded = F.upsampling_2d(deconv_5_decoded, cp.ones(deconv_5_decoded.shape), ksize=2, cover_all=False)
        # deconv_4_decoded = F.relu(self.decoder_conv_4(reshaped_decoded))
        up_4_decoded = F.upsampling_2d(reshaped_decoded, cp.ones(reshaped_decoded.shape), ksize=2, cover_all=False)
        # up_4_decoded = F.unpooling_2d(reshaped_decoded, ksize=2, cover_all=False)
        deconv_3_decoded = F.relu(self.decoder_conv_3(up_4_decoded))
        up_3_decoded = F.upsampling_2d(deconv_3_decoded, cp.ones(deconv_3_decoded.shape), ksize=2, cover_all=False)
        # up_3_decoded = F.unpooling_2d(deconv_3_decoded, ksize=2, cover_all=False)
        deconv_2_decoded = F.relu(self.decoder_conv_2(up_3_decoded))
        up_2_decoded = F.upsampling_2d(deconv_2_decoded, cp.ones(deconv_2_decoded.shape), ksize=2, cover_all=False)
        # up_2_decoded = F.unpooling_2d(deconv_2_decoded, ksize=2, cover_all=False)
        out_img = self.decoder_conv_1(up_2_decoded)
        # up_1_decoded = F.upsampling_2d(deconv_1_decoded, cp.ones(deconv_1_decoded.shape), ksize=2, cover_all=False)
        # out_img = self.decoder_conv_0(up_1_decoded)
    
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

    # def get_obj_labels(self, x):
    #     latent, _, _ = self.encode(x)
    #     labels = self.predict_obj_label(latent)

    #     return labels


    # predict the latent vector for each object
    def get_latent_indiv(self, x):
        latents = []
        mus = []
        ln_vars = []

        in_img = x

        offset_channels_n = self.rgb_channels_n + self.bg_channel_n + self.depth_channel_n
        for obj_idx in range(offset_channels_n, self.objects_n + offset_channels_n):
            rec_mask = in_img[:, obj_idx][:, None, :, :]

            # latent, mu, ln_var = self.encode(in_img[:, :3])
            latent, mu, ln_var = self.encode(F.concat((in_img[:, :4], rec_mask), axis=1))
            # latent, mu, ln_var = self.encode(rec_mask)

            mus.append(mu)
            ln_vars.append(ln_var)

            latents.append(latent)

        # mus = F.concat((mus), axis=1)
        # ln_vars = F.concat((ln_vars), axis=1)
        
        return latents, mus, ln_vars


    # predict the latent vector for the relationship
    def get_latent_rel(self, x):
        latents = []
        mus = []
        ln_vars = []

        latents, _, _ = self.get_latent_indiv(x)

        latent_concat = F.concat((latents), axis=1)

        for i in range(self.groups_rel_n):
            mu, ln_var = self.operators[i](latent_concat)
            mus.append(mu)
            ln_vars.append(ln_var)

        mus = F.concat((mus), axis=1)
        ln_vars = F.concat((ln_vars), axis=1)

        latent_rel = F.gaussian(mus, ln_vars)

        return latent_rel, mus, ln_vars


    def get_loss_func(self, k=1):

        def lf(x):

            in_img = x[0]
            if len(x) > 1:
                in_rel_labels = x[1]
                rel_masks = x[2]

                object_labels = x[3]
                object_label_masks = x[4]

            batchsize = float(len(in_img))
            denom = (k * batchsize * self.objects_n)

            # non_masked = [[] for _ in range(group_n)]
            # masks_flipped = [[] for _ in range(group_n)]

            # escape dividing by 0 when there are no labelled data points in the batch
            # for j in range(group_n):
            #     non_masked[j] = sum(cp.array(masks[j])) + 1
            #     masks_flipped[j] = 1 - cp.array(masks[j])

            
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
                    
                    # latents = []
                    # for _ in range(self.objects_n):
                    #     latents.append(F.gaussian(mus[obj_idx], ln_vars[obj_idx]))

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
                        in_spat_labels = object_labels[:, obj_idx, self.groups_obj_n:].astype(cp.float32)
                        masks = object_label_masks[:, obj_idx]

                        for i in range(self.groups_obj_n):
                        # for i in [1]:
                            # n = self.groups_len[i] - 1

                            # certain labels should not contribute to the calculation of the label loss values
                            # fixed_labels = (cp.tile(cp.array([1] + [-100] * n), (batchsize, 1)) * masks_flipped[i][:, cp.newaxis])
                            # out_labels[i] = out_labels[i] * cp.array(masks[i][:, cp.newaxis]) + fixed_labels

                            label_obj_acc += F.accuracy(out_obj_labels[i], cp.array(in_obj_labels[:, i])) / (k * self.objects_n)
                            label_obj_loss += F.softmax_cross_entropy(out_obj_labels[i], cp.array(in_obj_labels[:, i])) / denom

                        # for i in [-3, -2, -1]:
                        #     label_obj_loss += F.sum(F.squared_error(in_spat_labels[:, i], latents[0][:, i]))
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
                    # n = self.groups_len[i] - 1

                    # certain labels should not contribute to the calculation of the label loss values
                    # fixed_labels = (cp.tile(cp.array([1] + [-100] * n), (batchsize, 1)) * masks_flipped[i][:, cp.newaxis])
                    # out_rel_labels[i] = out_rel_labels[i] * cp.array(masks[i][:, cp.newaxis]) + fixed_labels

                    label_rel_acc += F.accuracy(out_rel_labels[i], cp.array(in_rel_labels[:, i])) / k
                    # label_rel_loss += F.softmax_cross_entropy(out_rel_labels[i], cp.array(in_rel_labels[:, i])) / (k * non_masked[i])
                    label_rel_loss += F.softmax_cross_entropy(out_rel_labels[i], cp.array(in_rel_labels[:, i])) / (k * batchsize)
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
                # self.loss += self.total_corr
            
            return self.loss, self.rec_loss, self.kl, self.label_obj_loss, self.label_obj_acc, self.label_rel_loss, self.label_rel_acc
        
        return lf

