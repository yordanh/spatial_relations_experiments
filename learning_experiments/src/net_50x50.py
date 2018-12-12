#!/usr/bin/env python
"""
title           :net.py
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

class Conv_Siam_VAE(chainer.Chain):
    """Convolutional Variational AutoEncoder"""

    def __init__(self, in_channels_branch_0=None, in_channels_branch_1=None, n_latent=None, groups=None, alpha=1, beta=100, gamma=100000):
        super(Conv_Siam_VAE, self).__init__()
        with self.init_scope():

            self.in_channels_branch_0 = in_channels_branch_0
            self.in_channels_branch_1 = in_channels_branch_1
            self.alpha = alpha
            self.beta = beta
            self.gamma= gamma
            self.n_latent = n_latent
            self.groups_len = [len(groups[key]) for key in sorted(groups.keys())]
            self.classifiers = chainer.ChainList()

            ########################
            # encoder for branch 0 #
            ########################
            self.encoder_conv_b0_0 = L.Convolution2D(in_channels_branch_0, 16, ksize=5, pad=2) # (50, 50)
            # max pool ksize=2 (25,25)
            self.encoder_conv_b0_1 = L.Convolution2D(16, 16, ksize=4) # (22, 22)
            # max pool ksize=2 (11,11)
            self.encoder_conv_b0_2 = L.Convolution2D(16, 8, ksize=4) # (8, 8)
            # reshape from (8, 8, 8) to (1,512)
            self.encoder_dense_b0_0 = L.Linear(512, 64)

            self.encoder_mu_b0 = L.Linear(64, self.n_latent)
            self.encoder_ln_var_b0 = L.Linear(64, self.n_latent)


            ########################
            # encoder for branch 1 #
            ########################
            self.encoder_conv_b1_0 = L.Convolution2D(in_channels_branch_1, 16, ksize=5, pad=2) # (50, 50)
            # max pool ksize=2 (25,25)
            self.encoder_conv_b1_1 = L.Convolution2D(16, 16, ksize=4) # (22, 22)
            # max pool ksize=2 (11,11)
            self.encoder_conv_b1_2 = L.Convolution2D(16, 8, ksize=4) # (8, 8)
            # reshape from (8, 8, 8) to (1,512)
            self.encoder_dense_b1_0 = L.Linear(512, 64)

            self.encoder_mu_b1 = L.Linear(64, self.n_latent)
            self.encoder_ln_var_b1 = L.Linear(64, self.n_latent)


            # label classifiers taking only the differences of the mean values into account
            for i in range(len(self.groups_len)):
                self.classifiers.add_link(L.Linear(1, self.groups_len[i]))

            self.operator_mu_0 = L.Linear(2 * self.n_latent, self.n_latent)
            self.operator_mu_1 = L.Linear(self.n_latent, self.n_latent)
            
            self.operator_ln_var_0 = L.Linear(2 * self.n_latent, self.n_latent)
            self.operator_ln_var_1 = L.Linear(self.n_latent, self.n_latent)


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
            self.decoder_conv_b0_2 = L.Convolution2D(16, 16, ksize=4) # (25, 25)
            # unpool ksize=2 (50, 50)
            self.decoder_output_img_b0 = L.Convolution2D(16, in_channels_branch_0, ksize=5, pad=2) # (50, 50)

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
            self.decoder_conv_b1_2 = L.Convolution2D(16, 16, ksize=4) # (25, 25)
            # unpool ksize=2 (50, 50)
            self.decoder_output_img_b1 = L.Convolution2D(16, in_channels_branch_1, ksize=5, pad=2) # (50, 50)


    def __call__(self, x_b0, x_b1):
        """AutoEncoder"""
        encoded = self.encode(x_b0, x_b1)
        return self.decode(encoded[2], encoded[4])

    def encode(self, x_b0, x_b1):
        conv_b0_0_encoded = F.relu(self.encoder_conv_b0_0(x_b0)) # (50, 50)
        pool_b0_0_encoded = F.max_pooling_2d(conv_b0_0_encoded, ksize=2) # (25, 25)
        conv_b0_1_encoded = F.relu(self.encoder_conv_b0_1(pool_b0_0_encoded)) # (22, 22)
        pool_b0_1_encoded = F.max_pooling_2d(conv_b0_1_encoded, ksize=2) # (11, 11)
        conv_b0_2_encoded = F.relu(self.encoder_conv_b0_2(pool_b0_1_encoded)) # (8, 8)
        reshaped_b0_encoded = F.reshape(conv_b0_2_encoded, (len(conv_b0_2_encoded), 1, 512)) # (1, 512)
        dense_b0_0_encoded = self.encoder_dense_b0_0(reshaped_b0_encoded) # (1, 8)
        mu_b0 = self.encoder_mu_b0(dense_b0_0_encoded) # (1, 2)
        ln_var_b0 = self.encoder_ln_var_b0(dense_b0_0_encoded)  # (1, 2) log(sigma**2)

        conv_b1_0_encoded = F.relu(self.encoder_conv_b1_0(x_b1)) # (50, 50)
        pool_b1_0_encoded = F.max_pooling_2d(conv_b1_0_encoded, ksize=2) # (25, 25)
        conv_b1_1_encoded = F.relu(self.encoder_conv_b1_1(pool_b1_0_encoded)) # (22, 22)
        pool_b1_1_encoded = F.max_pooling_2d(conv_b1_1_encoded, ksize=2) # (11, 11)
        conv_b1_2_encoded = F.relu(self.encoder_conv_b1_2(pool_b1_1_encoded)) # (8, 8)
        reshaped_b1_encoded = F.reshape(conv_b1_2_encoded, (len(conv_b1_2_encoded), 1, 512)) # (1, 512)
        dense_b1_0_encoded = self.encoder_dense_b1_0(reshaped_b1_encoded) # (1, 8)
        mu_b1 = self.encoder_mu_b1(dense_b1_0_encoded) # (1, 2)
        ln_var_b1 = self.encoder_ln_var_b1(dense_b1_0_encoded)  # (1, 2) log(sigma**2)

        # subtraction is the operator over the two latent vectors (vector per object)
        # mu = F.add(mu_b0, -1 * mu_b1)
        # ln_var = F.add(ln_var_b0, -1 * ln_var_b1)

        # learn the operator over the two latent vectors (vector per object)
        concat_mu = F.concat((mu_b0, mu_b1), axis=1)
        concat_ln_var = F.concat((ln_var_b0, ln_var_b1), axis=1)

        # mu = L.Linear(2 * self.n_latent, self.n_latent)(concat_mu)
        # ln_var = L.Linear(2 * self.n_latent, self.n_latent)(concat_ln_var)
        mu_tmp = F.relu(self.operator_mu_0(concat_mu))
        ln_var_tmp = F.relu(self.operator_ln_var_0(concat_ln_var))

        mu = self.operator_mu_1(mu_tmp)
        ln_var = self.operator_ln_var_1(ln_var_tmp)

        return mu, ln_var, mu_b0, ln_var_b0, mu_b1, ln_var_b1
        

    def decode(self, z_b0, z_b1, sigmoid=True):
        dense_b0_0_decoded = self.decoder_dense_b0_0(z_b0) # (1, 8)
        dense_b0_1_decoded = self.decoder_dense_b0_1(dense_b0_0_decoded) # (1, 512)
        reshapeb0_d_decoded = F.reshape(dense_b0_1_decoded, (len(dense_b0_1_decoded), 8, 8, 8))# (8, 8)
        deconv_b0_0_decoded = F.relu(self.decoder_conv_b0_0(reshapeb0_d_decoded)) # (8, 8)
        up_b0_0_decoded = F.unpooling_2d(deconv_b0_0_decoded, ksize=2, cover_all=False) # (16, 16)
        deconv_b0_1_decoded = F.relu(self.decoder_conv_b0_1(up_b0_0_decoded)) # (14, 14)
        up_b0_1_decoded = F.unpooling_2d(deconv_b0_1_decoded, ksize=2, cover_all=False) # (28, 28)
        deconv_b0_2_decoded = F.relu(self.decoder_conv_b0_2(up_b0_1_decoded)) # (25, 25)
        up_b0_2_decoded = F.unpooling_2d(deconv_b0_2_decoded, ksize=2, cover_all=False) # (50, 50)
        out_img_b0 = self.decoder_output_img_b0(up_b0_2_decoded) # (50, 50)

        dense_b1_0_decoded = self.decoder_dense_b1_0(z_b1) # (1, 8)
        dense_b1_1_decoded = self.decoder_dense_b1_1(dense_b1_0_decoded) # (1, 512)
        reshapeb1_d_decoded = F.reshape(dense_b1_1_decoded, (len(dense_b1_1_decoded), 8, 8, 8))# (8, 8)
        deconv_b1_0_decoded = F.relu(self.decoder_conv_b1_0(reshapeb1_d_decoded)) # (8, 8)
        up_b1_0_decoded = F.unpooling_2d(deconv_b1_0_decoded, ksize=2, cover_all=False) # (16, 16)
        deconv_b1_1_decoded = F.relu(self.decoder_conv_b1_1(up_b1_0_decoded)) # (14, 14)
        up_b1_1_decoded = F.unpooling_2d(deconv_b1_1_decoded, ksize=2, cover_all=False) # (28, 28)
        deconv_b1_2_decoded = F.relu(self.decoder_conv_b1_2(up_b1_1_decoded)) # (25, 25)
        up_b1_2_decoded = F.unpooling_2d(deconv_b1_2_decoded, ksize=2, cover_all=False) # (50, 50)
        out_img_b1 = self.decoder_output_img_b1(up_b1_2_decoded) # (50, 50)

        # need the check because the bernoulli_nll has a sigmoid in it
        if sigmoid:
            return F.sigmoid(out_img_b0), F.sigmoid(out_img_b1)
        else:
            return out_img_b0, out_img_b1
    
    def predict_label(self, zs, softmax=True):
        result = []

        # zs = F.gaussian(mus, ln_var)

        for i in range(len(self.groups_len)):
            z = zs[:, i, None]
            prediction = self.classifiers[i](z)

            # need the check because the softmax_cross_entropy has a softmax in it
            if softmax:
                result.append(F.softmax(prediction))
            else:
                result.append(prediction)

        return result

    def get_latent(self, x_b0, x_b1):
        mu, ln_var, _, _, _, _ = self.encode(x_b0, x_b1)
        return F.gaussian(mu, ln_var)

    def get_latent_mu(self, x_b0, x_b1):
        mu, ln_var, _, _, _, _ = self.encode(x_b0, x_b1)
        return mu

    def get_loss_func(self, k=1):
        """Get loss function of VAE."""

        def lf(x):

            group_n = len(self.groups_len)

            in_img_b0 = x[0]
            in_img_b1 = x[1]
            masks = x[2]
            in_labels = x[3]

            non_masked = []
            masks_flipped = []

            # escape dividing by 0 when there are no labelled data points in the batch
            for i in range(group_n):
                non_masked[i] = sum(masks[i]) + 1
                masks_flipped[i] = 1 - masks[i]

            rec_loss = 0
            label_loss = 0
            label_acc = 0
            # accs = [[],[]]

            mu, ln_var, mu_b0, ln_var_b0, mu_b1, ln_var_b1 = self.encode(in_img_b0, in_img_b1)
            batchsize = len(mu.data)

            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)

                out_img_b0, out_img_b1 = self.decode(mu_b0, mu_b1, sigmoid=False)
                rec_loss += F.bernoulli_nll(in_img_b0, out_img_b0) / (k * batchsize)
                rec_loss += F.bernoulli_nll(in_img_b1, out_img_b1) / (k * batchsize)

                # rec_sum = F.sum(F.bernoulli_nll(in_img, out_img, reduce='no') / (k * batchsize), axis=(1,2,3))
                # rec_sum += rec_sum * (cupy.array([1] * batchsize) * mask_flipped)
                # rec_loss += F.sum(rec_sum)
                

                out_labels = self.predict_label(z, softmax=False)
                for i in range(len(self.groups_len)):
                    n = self.groups_len[i] - 1

                    # certain labels should not contribute to the calculation of the label loss values
                    fixed_labels = (cupy.tile(cupy.array([1] + [-100] * n), (batchsize, 1)) * masks_flipped[i][:, cupy.newaxis])
                    out_labels[i] = out_labels[i] * masks[i][:, cupy.newaxis] + fixed_labels

                    label_acc_tmp = F.accuracy(out_labels[i], in_labels[i]) / k
                    label_acc += label_acc_tmp
                    # accs[i].append(label_acc_tmp) 
                    label_loss += self.gamma * F.softmax_cross_entropy(out_labels[i], in_labels[i]) / (k * non_masked[i])



            self.rec_loss = self.alpha * rec_loss
            self.label_loss = label_loss
            self.label_acc = label_acc

            kl = 0
            kl += 1 * gaussian_kl_divergence(mu[:,:2], ln_var[:,:2]) / (batchsize)
            kl += 1 * gaussian_kl_divergence(mu[:,2:], ln_var[:,2:]) / (batchsize)
            self.kl = self.beta * kl

            # self.mu_difference = 100 * F.mean(F.squared_difference(mu[:,:2], cupy.tile(cupy.array([0,0], dtype=cupy.float32), (batchsize, 1))))
            # # print(self.mu_difference)

            self.loss = self.rec_loss + self.label_loss + self.kl# + self.mu_difference
            
            return self.loss, self.rec_loss, self.label_loss, self.label_acc, self.kl
        
        return lf