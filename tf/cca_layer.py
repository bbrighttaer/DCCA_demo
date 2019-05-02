# Author: bbsipingsoft
# Project: DCCA_demo
# Date: 4/30/19
# Time: 11:31 PM
# File: cca_layer.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.engine.topology import Layer


class CCA(Layer):
    def __init__(self, out_dim=1, use_all_singular_vals=True, latent_dim=10, r1=1e-4, r2=1e-4, eps=1e-12, **kwargs):
        self.out_dim = out_dim
        self.latent_space_dim = latent_dim
        self.all_sing_vals = use_all_singular_vals
        self.r1 = tf.constant([r1], name='r1')
        self.r2 = tf.constant([r2], name='r2')
        self.eps = tf.constant([eps], name='epsilon')
        super(CCA, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CCA, self).build(input_shape)

    def call(self, inputs, **kwargs):
        o1 = o2 = tf.shape(inputs)[1] // 2

        H1 = K.transpose(inputs[:, :o1])
        H2 = K.transpose(inputs[:, o1: o1 + o2])

        one = tf.constant([1.0], name='one')
        m = tf.shape(H1)[1]
        m_float = tf.cast(m, 'float')

        # centered data matrices
        partition = tf.divide(one, tf.cast(m, 'float'), name='partition')
        H1_hat = H1 - partition * tf.matmul(H1, tf.ones((m, m)))
        H2_hat = H2 - partition * tf.matmul(H2, tf.ones((m, m)))

        sigma_partition = tf.divide(one, (m_float - 1), name='sigma_partition')
        SigmaHat12 = sigma_partition * tf.matmul(H1_hat, tf.transpose(H2_hat))
        SigmaHat11 = sigma_partition * tf.matmul(H1_hat, tf.transpose(H1_hat)) + self.r1 * tf.eye(o1)
        SigmaHat22 = sigma_partition * tf.matmul(H2_hat, tf.transpose(H2_hat)) + self.r2 * tf.eye(o2)

        # SVD decomposition for square root calculation
        [D1, V1] = tf.py_func(lambda x: np.linalg.eigh(x), [SigmaHat11], [tf.float32, tf.float32])
        [D2, V2] = tf.py_func(lambda x: np.linalg.eigh(x), [SigmaHat22], [tf.float32, tf.float32])

        D1_indices = tf.where(D1 > self.eps)
        D1_indices = tf.squeeze(D1_indices)
        V1 = tf.gather(V1, D1_indices)
        D1 = tf.gather(D1, D1_indices)

        D2_indices = tf.where(D2 > self.eps)
        D2_indices = tf.squeeze(D2_indices)
        V2 = tf.gather(V2, D2_indices)
        D2 = tf.gather(D2, D2_indices)

        # calculate root inverse of correlation matrices
        pow_val = tf.constant([-0.5])
        SigmaHat11RootInv = tf.matmul(V1, tf.matmul(tf.diag(tf.pow(D1, pow_val)), tf.transpose(V1)))
        SigmaHat22RootInv = tf.matmul(V2, tf.matmul(tf.diag(tf.pow(D2, pow_val)), tf.transpose(V2)))

        # Total correlation
        T = tf.matmul(SigmaHat11RootInv, tf.matmul(SigmaHat12, SigmaHat22RootInv))
        TT = tf.matmul(tf.transpose(T), T)
        if self.all_sing_vals:
            corr = K.sqrt(tf.trace(TT))
            # corr = tf.trace(K.sqrt(TT))
        else:
            eig_vals, eig_vecs = tf.linalg.eigh(TT)
            topk_eig_vals, _ = tf.nn.top_k(eig_vals, self.latent_space_dim)
            corr = tf.sqrt(tf.reduce_sum(topk_eig_vals))
            # corr = K.sum(K.sqrt(topk_eig_vals))

        return -corr

    def get_config(self):
        return super(CCA, self).get_config()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_dim)
