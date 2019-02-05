#!/usr/bin/env python
"""
title           :net_100x100.py
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
import cupy

import numpy as np


class Operator(chainer.Chain):
    """Fully-connected feedforwards operator"""

    def __init__(self, input_channels=None, n_latent=None, embed_size=None):
        super(Operator, self).__init__()
        with self.init_scope():

            self.n_latent = n_latent
            self.embed_size = embed_size
            
            self.operator_0 = L.Linear(2 * input_channels, self.n_latent)
            self.operator_1 = L.Linear(self.n_latent, self.n_latent)
            self.operator_mu = L.Linear(self.n_latent, self.embed_size)
            self.operator_ln_var = L.Linear(self.n_latent, self.embed_size)


    def __call__(self, z_concat):
        """AutoEncoder"""
        z = self.encode(z_concat)
        return z

    def encode(self, z_concat):

        # z_tmp = F.leaky_relu(self.operator_0(z_concat))
        z_tmp0 = F.leaky_relu(self.operator_0(z_concat))
        z_tmp1 = F.leaky_relu(self.operator_1(z_tmp0))
        mu = self.operator_mu(z_tmp1)
        ln_var = self.operator_ln_var(z_tmp1)
        return mu, ln_var



class Conv_Siam_VAE(chainer.Chain):
    """Convolutional Variational AutoEncoder"""

    def __init__(self, in_channels_n=None, n_latent=None, groups=None, alpha=1, beta=100, gamma=100000):
        super(Conv_Siam_VAE, self).__init__()
        with self.init_scope():

            self.in_channels_n = in_channels_n
            self.alpha = alpha
            self.beta = beta
            self.gamma= gamma
            self.n_latent = n_latent
            self.groups_len = [len(groups[key]) for key in sorted(groups.keys())]
            self.group_n = len(self.groups_len)
            self.classifiers = chainer.ChainList()
            self.operators = chainer.ChainList()

            ########################
            # encoder for branch 0 #
            ######################## 
            self.encoder_conv_b0_0 = L.Convolution2D(self.in_channels_n, 32, ksize=7, pad=3) # (100, 100)
            # max pool ksize=2 (50, 50)
            self.encoder_conv_b0_1 = L.Convolution2D(32, 32, ksize=5, pad=2) # (50, 50)
            # max pool ksize=2 (25,25)
            self.encoder_conv_b0_2 = L.Convolution2D(32, 16, ksize=4) # (22, 22)
            # max pool ksize=2 (11,11)
            self.encoder_conv_b0_3 = L.Convolution2D(16, 8, ksize=4) # (8, 8)
            # reshape from (8, 8, 8) to (1,512)
            self.encoder_dense_b0_0 = L.Linear(512, 64)

            self.encoder_mu_b0 = L.Linear(64, self.n_latent)
            self.encoder_ln_var_b0 = L.Linear(64, self.n_latent)

            # label classifiers taking the prodices values by the operators
            # for each concept group
            # each operator takes the concatenated samples from the latent space
            # of each branch
            for i in range(self.group_n):
                self.classifiers.add_link(L.Linear(1, self.groups_len[i]))
                self.operators.add_link(Operator(input_channels=self.group_n, n_latent=self.n_latent, embed_size=1))


            ########################
            # decoder for branch 0 #
            ########################
            self.decoder_dense_b0_0 = L.Linear(self.n_latent, 64)
            self.decoder_dense_b0_1 = L.Linear(64, 512)
            # reshape from (1, 512) to (8, 8, 8)
            self.decoder_conv_b0_0 = L.Convolution2D(8, 8, ksize=3, pad=1) # (8, 8)
            # unpool ksize=2 (16, 16)
            self.decoder_conv_b0_1 = L.Convolution2D(8, 16, ksize=3) # (14, 14)
            # unpool ksize=2 (28, 28)
            self.decoder_conv_b0_2 = L.Convolution2D(16, 32, ksize=4) # (25, 25)
            # unpool ksize=2 (50, 50)
            self.decoder_conv_b0_3 = L.Convolution2D(32, 32, ksize=5, pad=2) # (50, 50)
            # unpool ksize=2 (100, 100)
            self.decoder_output_img_b0 = L.Convolution2D(32, self.in_channels_n, ksize=7, pad=3) # (100, 100)

            ########################
            # decoder for branch 1 #
            ########################
            self.decoder_dense_b1_0 = L.Linear(self.n_latent, 64)
            self.decoder_dense_b1_1 = L.Linear(64, 512)
            # reshape from (1, 512) to (8, 8, 8)
            self.decoder_conv_b1_0 = L.Convolution2D(8, 8, ksize=3, pad=1) # (8, 8)
            # unpool ksize=2 (16, 16)
            self.decoder_conv_b1_1 = L.Convolution2D(8, 16, ksize=3) # (14, 14)
            # unpool ksize=2 (28, 28)
            self.decoder_conv_b1_2 = L.Convolution2D(16, 32, ksize=4) # (25, 25)
            # unpool ksize=2 (50, 50)
            self.decoder_conv_b1_3 = L.Convolution2D(32, 32, ksize=5, pad=2) # (50, 50)
             # unpool ksize=2 (100, 100)
            self.decoder_output_img_b1 = L.Convolution2D(32, self.in_channels_n, ksize=7, pad=3) # (100, 100)


    def __call__(self, x_b0, x_b1):
        encoded = self.encode(x_b0, x_b1)
        mu_b0 = encoded[2]
        ln_var_b0 = encoded[3]

        mu_b1 = encoded[4]
        ln_var_b1 = encoded[5]

        z_b0_sample = F.gaussian(mu_b0[:, self.group_n:], ln_var_b0[:, self.group_n:])
        z_b1_sample = F.gaussian(mu_b1[:, self.group_n:], ln_var_b1[:, self.group_n:]) 

        _, latent, _, _ = self.predict_label(mu_b0, mu_b1)
        out_labels, latent, mus, ln_vars = self.predict_label(mu_b1, mu_b0)
        return self.decode(mu_b0[:, self.group_n:], mu_b1[:, self.group_n:], latent)

    def encode(self, x_b0, x_b1):
        conv_b0_0_encoded = F.leaky_relu(self.encoder_conv_b0_0(x_b0))# (100, 100)
        pool_b0_0_encoded = F.max_pooling_2d(conv_b0_0_encoded, ksize=2) # (50, 50)
        conv_b0_1_encoded = F.leaky_relu(self.encoder_conv_b0_1(pool_b0_0_encoded)) # (50, 50)
        pool_b0_1_encoded = F.max_pooling_2d(conv_b0_1_encoded, ksize=2) # (25, 25)
        conv_b0_2_encoded = F.leaky_relu(self.encoder_conv_b0_2(pool_b0_1_encoded)) # (22, 22)
        pool_b0_2_encoded = F.max_pooling_2d(conv_b0_2_encoded, ksize=2) # (11, 11)
        conv_b0_3_encoded = F.leaky_relu(self.encoder_conv_b0_3(pool_b0_2_encoded)) # (8, 8)
        reshaped_b0_encoded = F.reshape(conv_b0_3_encoded, (len(conv_b0_3_encoded), 1, 512)) # (1, 512)
        dense_b0_0_encoded = F.leaky_relu(self.encoder_dense_b0_0(reshaped_b0_encoded)) # (1, 8)
        mu_b0 = self.encoder_mu_b0(dense_b0_0_encoded) # (1, 2)
        ln_var_b0 = self.encoder_ln_var_b0(dense_b0_0_encoded)  # (1, 2) log(sigma**2)


        conv_b1_0_encoded = F.leaky_relu(self.encoder_conv_b0_0(x_b1)) # (100, 100)
        pool_b1_0_encoded = F.max_pooling_2d(conv_b1_0_encoded, ksize=2) # (50, 50)
        conv_b1_1_encoded = F.leaky_relu(self.encoder_conv_b0_1(pool_b1_0_encoded)) # (50, 50)
        pool_b1_1_encoded = F.max_pooling_2d(conv_b1_1_encoded, ksize=2) # (25, 25)
        conv_b1_2_encoded = F.leaky_relu(self.encoder_conv_b0_2(pool_b1_1_encoded)) # (22, 22)
        pool_b1_2_encoded = F.max_pooling_2d(conv_b1_2_encoded, ksize=2) # (11, 11)
        conv_b1_3_encoded = F.leaky_relu(self.encoder_conv_b0_3(pool_b1_2_encoded)) # (8, 8)
        reshaped_b1_encoded = F.reshape(conv_b1_3_encoded, (len(conv_b1_3_encoded), 1, 512)) # (1, 512)
        dense_b1_0_encoded = F.leaky_relu(self.encoder_dense_b0_0(reshaped_b1_encoded)) # (1, 8)
        mu_b1 = self.encoder_mu_b0(dense_b1_0_encoded) # (1, 2)
        ln_var_b1 = self.encoder_ln_var_b0(dense_b1_0_encoded)  # (1, 2) log(sigma**2)

        return F.gaussian(mu_b0, ln_var_b0), F.gaussian(mu_b1, ln_var_b1), mu_b0, ln_var_b0, mu_b1, ln_var_b1
        

    def decode(self, z_b0, z_b1, latent, sigmoid=True):

        concat_b0 = F.concat((latent, z_b0), axis=1)
        concat_b1 = F.concat((latent, z_b1), axis=1)

        dense_b0_0_decoded = self.decoder_dense_b0_0(concat_b0) # (1, 10)
        dense_b0_1_decoded = F.leaky_relu(self.decoder_dense_b0_1(dense_b0_0_decoded)) # (1, 512)
        reshapeb0_d_decoded = F.reshape(dense_b0_1_decoded, (len(dense_b0_1_decoded), 8, 8, 8))# (8, 8)
        deconv_b0_0_decoded = F.leaky_relu(self.decoder_conv_b0_0(reshapeb0_d_decoded)) # (8, 8)
        up_b0_0_decoded = F.unpooling_2d(deconv_b0_0_decoded, ksize=2, cover_all=False) # (16, 16)
        deconv_b0_1_decoded = F.leaky_relu(self.decoder_conv_b0_1(up_b0_0_decoded)) # (14, 14)
        up_b0_1_decoded = F.unpooling_2d(deconv_b0_1_decoded, ksize=2, cover_all=False) # (28, 28)
        deconv_b0_2_decoded = F.leaky_relu(self.decoder_conv_b0_2(up_b0_1_decoded)) # (25, 25)
        up_b0_2_decoded = F.unpooling_2d(deconv_b0_2_decoded, ksize=2, cover_all=False) # (50, 50)
        deconv_b0_3_decoded = F.leaky_relu(self.decoder_conv_b0_3(up_b0_2_decoded)) # (50, 50)
        up_b0_3_decoded = F.unpooling_2d(deconv_b0_3_decoded, ksize=2, cover_all=False) # (100, 100)
        out_img_b0 = self.decoder_output_img_b0(up_b0_3_decoded) # (100, 100)

        # takes the encoding from the first branch and the spatial embedding as input
        # to recreate the pointcloud of the second branch
        dense_b1_0_decoded = self.decoder_dense_b1_0(concat_b1) # (1, 8)
        dense_b1_1_decoded = F.leaky_relu(self.decoder_dense_b1_1(dense_b1_0_decoded)) # (1, 512)
        reshapeb1_d_decoded = F.reshape(dense_b1_1_decoded, (len(dense_b1_1_decoded), 8, 8, 8))# (8, 8)
        deconv_b1_0_decoded = F.leaky_relu(self.decoder_conv_b1_0(reshapeb1_d_decoded)) # (8, 8)
        up_b1_0_decoded = F.unpooling_2d(deconv_b1_0_decoded, ksize=2, cover_all=False) # (16, 16)
        deconv_b1_1_decoded = F.leaky_relu(self.decoder_conv_b1_1(up_b1_0_decoded)) # (14, 14)
        up_b1_1_decoded = F.unpooling_2d(deconv_b1_1_decoded, ksize=2, cover_all=False) # (28, 28)
        deconv_b1_2_decoded = F.leaky_relu(self.decoder_conv_b1_2(up_b1_1_decoded)) # (25, 25)
        up_b1_2_decoded = F.unpooling_2d(deconv_b1_2_decoded, ksize=2, cover_all=False) # (50, 50)
        deconv_b1_3_decoded = F.leaky_relu(self.decoder_conv_b1_3(up_b1_2_decoded)) # (50, 50)
        up_b1_3_decoded = F.unpooling_2d(deconv_b1_3_decoded, ksize=2, cover_all=False) # (100, 100)
        out_img_b1 = self.decoder_output_img_b1(up_b1_3_decoded) # (100, 100)


        # need the check because the bernoulli_nll has a sigmoid in it
        if sigmoid:
            return F.sigmoid(out_img_b0), F.sigmoid(out_img_b1)
        else:
            return out_img_b0, out_img_b1
    
    def predict_label(self, z_b0, z_b1, softmax=True):
        result = []
        latents = []
        mus = []
        ln_vars = []

        z_concat = F.concat((z_b0[:, :self.group_n], z_b1[:, :self.group_n]), axis=1)

        for i in range(self.group_n):

            mu, ln_var = self.operators[i](z_concat)

            mus.append(mu)
            ln_vars.append(ln_var)

        mus_return = F.concat((mus), axis=1)
        ln_vars_return = F.concat((ln_vars), axis=1)

        latent = F.gaussian(mus_return, ln_vars_return)
        for i in range(self.group_n):
            prediction = self.classifiers[i](latent[:, i, None])

            # need the check because the softmax_cross_entropy has a softmax in it
            if softmax:
                result.append(F.softmax(prediction))
            else:
                result.append(prediction)

        return result, latent, mus_return, ln_vars_return


    def get_latent(self, x_b0, x_b1):
        z_b0, z_b1, mu_b0, _, mu_b1, _ = self.encode(x_b0, x_b1)
        _, latent, _, _ = self.predict_label(mu_b0, mu_b1)

        return latent


    def get_latent_pred(self, x_b0, x_b1):
        z_b0, z_b1, mu_b0, _, mu_b1, _ = self.encode(x_b0, x_b1)
        _, _, mus, ln_vars = self.predict_label(mu_b0, mu_b1)

        return mus, ln_vars


    def get_label(self, x_b0, x_b1):
        z_b0, z_b1, mu_b0, _, mu_b1, _ = self.encode(x_b0, x_b1)
        labels, _, _, _ = self.predict_label(mu_b0, mu_b1)

        return labels


    def get_loss_func(self, k=1):

        def lf(x):

            group_n = len(self.groups_len)
            in_img_b0 = x[0]
            in_img_b1 = x[1]
            masks = x[2 : 2 + group_n]
            in_labels = x[2 + group_n : ]

            non_masked = [[] for _ in range(group_n)]
            masks_flipped = [[] for _ in range(group_n)]

            # escape dividing by 0 when there are no labelled data points in the batch
            for j in range(group_n):
                non_masked[j] = sum(masks[j]) + 1
                masks_flipped[j] = 1 - masks[j]

            rec_loss = 0
            label_loss = 0
            label_acc = 0

            z_b0, z_b1, mu_b0, ln_var_b0, mu_b1, ln_var_b1 = self.encode(in_img_b0, in_img_b1)
            batchsize = len(z_b0.data)

            for l in six.moves.range(k):               

                out_labels, latent, mus, ln_vars = self.predict_label(mu_b0, mu_b1, softmax=False)

                out_img_b0, out_img_b1 = self.decode(mu_b0[:, self.group_n:], mu_b1[:, self.group_n:], latent, sigmoid=False)

                rec_loss += F.bernoulli_nll(in_img_b0, out_img_b0) / (k * batchsize)
                rec_loss += F.bernoulli_nll(in_img_b1, out_img_b1) / (k * batchsize)

                for i in range(self.group_n):
                    n = self.groups_len[i] - 1

                    # certain labels should not contribute to the calculation of the label loss values
                    fixed_labels = (cupy.tile(cupy.array([1] + [-100] * n), (batchsize, 1)) * masks_flipped[i][:, cupy.newaxis])
                    out_labels[i] = out_labels[i] * masks[i][:, cupy.newaxis] + fixed_labels

                    label_acc_tmp = F.accuracy(out_labels[i], in_labels[i]) / k
                    label_acc += label_acc_tmp
                    label_loss += F.softmax_cross_entropy(out_labels[i], in_labels[i]) / (k * non_masked[i])



            self.rec_loss = self.alpha * rec_loss
            self.label_loss = self.gamma * label_loss
            self.label_acc = label_acc

            kl = 0
            kl += gaussian_kl_divergence(mus, ln_vars) / (batchsize)

            self.kl = self.beta * kl

            self.loss = self.rec_loss + self.label_loss + self.kl
            
            return self.loss, self.rec_loss, self.label_loss, self.label_acc, self.kl
        
        return lf



class Conv_Siam_Classifier(Conv_Siam_VAE):
    """Convolutional Variational AutoEncoder"""

    def __init__(self, in_channels_n=None, n_latent=None, groups=None, alpha=1, beta=100, gamma=100000):
        super(Conv_Siam_Classifier, self).__init__(in_channels_n=in_channels_n, 
                                                   n_latent=n_latent, groups=groups, 
                                                   alpha=alpha, beta=beta, gamma=gamma)
    
    def predict_label(self, z_b0, z_b1, softmax=True):
        result = []
        latents = []
        mus = []
        lns = []

        z_concat = F.concat((z_b0[:, :self.group_n], z_b1[:, :self.group_n]), axis=1)

        for i in range(self.group_n):

            mu, ln = self.operators[i](z_concat)

            mus.append(mu)
            lns.append(ln)

        mus_return = F.concat((mus), axis=1)
        lns_returns = F.concat((lns), axis=1)

        latent = mus_return
        for i in range(self.group_n):
            prediction = self.classifiers[i](latent[:, i, None])

            # need the check because the softmax_cross_entropy has a softmax in it
            if softmax:
                result.append(F.softmax(prediction))
            else:
                result.append(prediction)

        return result, latent, mus_return, lns_returns



class Conv_Siam_BetaVAE(Conv_Siam_VAE):
    """Convolutional Variational AutoEncoder"""

    def __init__(self, in_channels_n=None, n_latent=None, groups=None, alpha=1, beta=100, gamma=100000):
        super(Conv_Siam_BetaVAE, self).__init__(in_channels_n=in_channels_n, 
                                                n_latent=n_latent, groups=groups, 
                                                alpha=alpha, beta=beta, gamma=gamma)

        self.operators = chainer.ChainList()
        self.operators.add_link(Operator(input_channels=self.n_latent, n_latent=self.n_latent, embed_size=self.n_latent))
    

    def decode(self, z_b0, z_b1, latent, sigmoid=True):

        dense_b0_0_decoded = self.decoder_dense_b0_0(latent) # (1, 8)
        dense_b0_1_decoded = F.leaky_relu(self.decoder_dense_b0_1(dense_b0_0_decoded)) # (1, 512)
        reshapeb0_d_decoded = F.reshape(dense_b0_1_decoded, (len(dense_b0_1_decoded), 8, 8, 8))# (8, 8)
        deconv_b0_0_decoded = F.leaky_relu(self.decoder_conv_b0_0(reshapeb0_d_decoded)) # (8, 8)
        up_b0_0_decoded = F.unpooling_2d(deconv_b0_0_decoded, ksize=2, cover_all=False) # (16, 16)
        deconv_b0_1_decoded = F.leaky_relu(self.decoder_conv_b0_1(up_b0_0_decoded)) # (14, 14)
        up_b0_1_decoded = F.unpooling_2d(deconv_b0_1_decoded, ksize=2, cover_all=False) # (28, 28)
        deconv_b0_2_decoded = F.leaky_relu(self.decoder_conv_b0_2(up_b0_1_decoded)) # (25, 25)
        up_b0_2_decoded = F.unpooling_2d(deconv_b0_2_decoded, ksize=2, cover_all=False) # (50, 50)
        deconv_b0_3_decoded = F.leaky_relu(self.decoder_conv_b0_3(up_b0_2_decoded)) # (50, 50)
        up_b0_3_decoded = F.unpooling_2d(deconv_b0_3_decoded, ksize=2, cover_all=False) # (100, 100)
        out_img_b0 = self.decoder_output_img_b0(up_b0_3_decoded) # (100, 100)

        # takes the encoding from the first branch and the spatial embedding as input
        # to recreate the pointcloud of the second branch
        dense_b1_0_decoded = self.decoder_dense_b1_0(latent) # (1, 8)
        dense_b1_1_decoded = F.leaky_relu(self.decoder_dense_b1_1(dense_b1_0_decoded)) # (1, 512)
        reshapeb1_d_decoded = F.reshape(dense_b1_1_decoded, (len(dense_b1_1_decoded), 8, 8, 8))# (8, 8)
        deconv_b1_0_decoded = F.leaky_relu(self.decoder_conv_b1_0(reshapeb1_d_decoded)) # (8, 8)
        up_b1_0_decoded = F.unpooling_2d(deconv_b1_0_decoded, ksize=2, cover_all=False) # (16, 16)
        deconv_b1_1_decoded = F.leaky_relu(self.decoder_conv_b1_1(up_b1_0_decoded)) # (14, 14)
        up_b1_1_decoded = F.unpooling_2d(deconv_b1_1_decoded, ksize=2, cover_all=False) # (28, 28)
        deconv_b1_2_decoded = F.leaky_relu(self.decoder_conv_b1_2(up_b1_1_decoded)) # (25, 25)
        up_b1_2_decoded = F.unpooling_2d(deconv_b1_2_decoded, ksize=2, cover_all=False) # (50, 50)
        deconv_b1_3_decoded = F.leaky_relu(self.decoder_conv_b1_3(up_b1_2_decoded)) # (50, 50)
        up_b1_3_decoded = F.unpooling_2d(deconv_b1_3_decoded, ksize=2, cover_all=False) # (100, 100)
        out_img_b1 = self.decoder_output_img_b1(up_b1_3_decoded) # (100, 100)

        # need the check because the bernoulli_nll has a sigmoid in it
        if sigmoid:
            return F.sigmoid(out_img_b0), F.sigmoid(out_img_b1)
        else:
            return out_img_b0, out_img_b1


    def predict_label(self, z_b0, z_b1, softmax=True):
        result = []
        latents = []
        mus = []
        ln_vars = []

        z_concat = F.concat((z_b0, z_b1), axis=1)

        mu, ln_var = self.operators[-1](z_concat)

        latent = F.gaussian(mu, ln_var)
        
        for i in range(self.group_n):
            prediction = self.classifiers[i](latent[:, i, None])

            # need the check because the softmax_cross_entropy has a softmax in it
            if softmax:
                result.append(F.softmax(prediction))
            else:
                result.append(prediction)

        return result, latent, mu, ln_var



class Conv_Siam_AE(Conv_Siam_VAE):
    """Convolutional Variational AutoEncoder"""

    def __init__(self, in_channels_n=None, n_latent=None, groups=None, alpha=1, beta=100, gamma=100000):
        super(Conv_Siam_AE, self).__init__(in_channels_n=in_channels_n, 
                                           n_latent=n_latent, groups=groups, 
                                           alpha=alpha, beta=beta, gamma=gamma)
    
        self.operators = chainer.ChainList()
        self.operators.add_link(Operator(input_channels=self.n_latent, n_latent=self.n_latent, embed_size=self.n_latent))

    def decode(self, z_b0, z_b1, latent, sigmoid=True):

        dense_b0_0_decoded = self.decoder_dense_b0_0(latent) # (1, 8)
        dense_b0_1_decoded = F.leaky_relu(self.decoder_dense_b0_1(dense_b0_0_decoded)) # (1, 512)
        reshapeb0_d_decoded = F.reshape(dense_b0_1_decoded, (len(dense_b0_1_decoded), 8, 8, 8))# (8, 8)
        deconv_b0_0_decoded = F.leaky_relu(self.decoder_conv_b0_0(reshapeb0_d_decoded)) # (8, 8)
        up_b0_0_decoded = F.unpooling_2d(deconv_b0_0_decoded, ksize=2, cover_all=False) # (16, 16)
        deconv_b0_1_decoded = F.leaky_relu(self.decoder_conv_b0_1(up_b0_0_decoded)) # (14, 14)
        up_b0_1_decoded = F.unpooling_2d(deconv_b0_1_decoded, ksize=2, cover_all=False) # (28, 28)
        deconv_b0_2_decoded = F.leaky_relu(self.decoder_conv_b0_2(up_b0_1_decoded)) # (25, 25)
        up_b0_2_decoded = F.unpooling_2d(deconv_b0_2_decoded, ksize=2, cover_all=False) # (50, 50)
        deconv_b0_3_decoded = F.leaky_relu(self.decoder_conv_b0_3(up_b0_2_decoded)) # (50, 50)
        up_b0_3_decoded = F.unpooling_2d(deconv_b0_3_decoded, ksize=2, cover_all=False) # (100, 100)
        out_img_b0 = self.decoder_output_img_b0(up_b0_3_decoded) # (100, 100)

        # takes the encoding from the first branch and the spatial embedding as input
        # to recreate the pointcloud of the second branch
        dense_b1_0_decoded = self.decoder_dense_b1_0(latent) # (1, 8)
        dense_b1_1_decoded = F.leaky_relu(self.decoder_dense_b1_1(dense_b1_0_decoded)) # (1, 512)
        reshapeb1_d_decoded = F.reshape(dense_b1_1_decoded, (len(dense_b1_1_decoded), 8, 8, 8))# (8, 8)
        deconv_b1_0_decoded = F.leaky_relu(self.decoder_conv_b1_0(reshapeb1_d_decoded)) # (8, 8)
        up_b1_0_decoded = F.unpooling_2d(deconv_b1_0_decoded, ksize=2, cover_all=False) # (16, 16)
        deconv_b1_1_decoded = F.leaky_relu(self.decoder_conv_b1_1(up_b1_0_decoded)) # (14, 14)
        up_b1_1_decoded = F.unpooling_2d(deconv_b1_1_decoded, ksize=2, cover_all=False) # (28, 28)
        deconv_b1_2_decoded = F.leaky_relu(self.decoder_conv_b1_2(up_b1_1_decoded)) # (25, 25)
        up_b1_2_decoded = F.unpooling_2d(deconv_b1_2_decoded, ksize=2, cover_all=False) # (50, 50)
        deconv_b1_3_decoded = F.leaky_relu(self.decoder_conv_b1_3(up_b1_2_decoded)) # (50, 50)
        up_b1_3_decoded = F.unpooling_2d(deconv_b1_3_decoded, ksize=2, cover_all=False) # (100, 100)
        out_img_b1 = self.decoder_output_img_b1(up_b1_3_decoded) # (100, 100)

        # need the check because the bernoulli_nll has a sigmoid in it
        if sigmoid:
            return F.sigmoid(out_img_b0), F.sigmoid(out_img_b1)
        else:
            return out_img_b0, out_img_b1


    def predict_label(self, z_b0, z_b1, softmax=True):
        result = []
        latents = []
        mus = []
        ln_vars = []

        z_concat = F.concat((z_b0, z_b1), axis=1)

        mu, _ = self.operators[-1](z_concat)

        latent = mu
        
        for i in range(self.group_n):
            prediction = self.classifiers[i](latent[:, i, None])

            # need the check because the softmax_cross_entropy has a softmax in it
            if softmax:
                result.append(F.softmax(prediction))
            else:
                result.append(prediction)

        return result, latent, _, _