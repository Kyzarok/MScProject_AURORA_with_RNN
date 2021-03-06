from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.examples.tutorials.mnist import input_data

from my_nn_lib import Convolution2D, MaxPooling2D, Conv2Dtranspose
from my_nn_lib import FullConnected, ReadOutLayer


class AE(object):
    
    def __init__(self):
        self.traj_length = 50;
        self.x = tf.placeholder(tf.float32, [None, self.traj_length*2], name="input_x")

        self.x_image = tf.reshape(self.x, [-1, 2, self.traj_length, 1])
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.latent_dim = 2
        
        self.global_step = tf.placeholder(tf.int32, shape=(), name="step_id")

        self.create_net()
        self.create_loss()
        self.create_optimizer()
        self.saver = tf.train.Saver(tf.trainable_variables())

    def create_optimizer(self):
        self.learning_rate = tf.train.exponential_decay(0.1, self.global_step,
                                                        250000, 0.9,name="learning_rate")
        optimizer=tf.train.AdagradOptimizer(self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_step = optimizer.apply_gradients(zip(gradients, variables), name="train_step")
        
        self.reset_optimizer_op = tf.variables_initializer(optimizer.variables(), name="reset_optimizer")

    def create_net(self):
        self.layers=[self.x_image]
        with tf.variable_scope('encoder') as vs:
            self.create_encoder_conv([2])
            self.create_encoder_fc([5]) 

            self.latent = FullConnected(self.layers[-1], self.layers[-1].get_shape().as_list()[1], self.latent_dim, activation='identity', name = "latent").output()
            self.layers.append(self.latent)

        with tf.variable_scope('decoder') as vs:
            self.create_decoder_fc([5, self.traj_length * 2],[-1, self.traj_length, 2, 1])
            self.create_decoder_conv([2,1])
            
            print("Last layer")
            print (self.layers[-1].get_shape().as_list())
            res=tf.reshape(self.layers[-1], [-1, self.traj_length*2], name = "reconstructed")
            self.layers.append(res)
            self.decoded  = FullConnected(self.layers[-1], self.layers[-1].get_shape().as_list()[1], 100, activation = 'identity', name = "decoded").output()
            self.layers.append(self.decoded)

    def create_loss(self):
        recon_loss = math_ops.squared_difference(self.x,self.decoded, name = "recon_loss")
        self.loss = tf.reduce_mean(recon_loss, 0, name = "loss")


    def create_encoder_fc(self, conf):
        size=self.layers[-1].get_shape().as_list()
        self.layers.append(tf.reshape(self.layers[-1], [-1, size[1]*size[2]*size[3] ]))
        for i in range(len(conf)):
            if i>1:
                drop = tf.nn.dropout(self.layers[-1], self.keep_prob)
                self.layers.append(drop)

            fc = FullConnected(self.layers[-1], self.layers[-1].get_shape().as_list()[1], conf[i], activation='sigmoid')
            self.layers.append(fc.output())

    def create_decoder_fc(self, conf, final_shape):
        for i in range(len(conf)):
            if i>1:
                drop = tf.nn.dropout(self.layers[-1], self.keep_prob)
                self.layers.append(drop)

            fc = FullConnected(self.layers[-1], self.layers[-1].get_shape().as_list()[1], conf[i], activation='sigmoid')
            self.layers.append(fc.output())
        self.layers.append(tf.reshape(self.layers[-1], final_shape))
                                    
        
    def create_encoder_conv(self, conf):
        for i in range(len(conf)):
            conv = Convolution2D(self.layers[-1], (self.layers[-1].get_shape().as_list()[1], self.layers[-1].get_shape().as_list()[2]), self.layers[-1].get_shape().as_list()[3], conf[i],(2, 6), activation='leaky_relu')
            self.layers.append(conv.output())
            pool = MaxPooling2D(self.layers[-1])
            self.layers.append(pool.output())

    def create_decoder_conv(self, conf):
        for i in range(len(conf)):
            conv_t = Conv2Dtranspose(self.layers[-1],(self.layers[-1].get_shape().as_list()[1], self.layers[-1].get_shape().as_list()[2]), self.layers[-1].get_shape().as_list()[3], conf[i],
                                     (2, 6), activation='leak_relu')
            self.layers.append(conv_t.output())



