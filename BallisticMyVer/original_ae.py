from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.python.ops import math_ops

from original_my_nn_lib import Convolution2D, MaxPooling2D, Conv2Dtranspose
from original_my_nn_lib import FullConnected, ReadOutLayer
from original_my_nn_lib import LSTM_layer


class AE(object):
    
    def __init__(self, with_rnn):
        # Same settings for both networks
        self.traj_length = 50
        self.latent_dim = 2
        self.global_step = tf.placeholder(tf.int32, shape=(), name="step_id")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        if with_rnn == False:
            self.x = tf.placeholder(tf.float32, [None, self.traj_length*2], name="input_x")
            self.x_image = tf.reshape(self.x, [-1, 2, self.traj_length, 1])
        else:
            self.x = tf.placeholder(tf.float32, [None, self.traj_length], name="rnn_input")
            self.true_x = tf.placeholder(tf.float32, [None, self.traj_length*2], name="real_x")

        self.create_net(with_rnn)

        self.create_loss(with_rnn)
        self.create_optimizer()
        self.saver = tf.train.Saver(tf.trainable_variables())

    def create_optimizer(self):
        self.learning_rate = tf.train.exponential_decay(0.1, self.global_step, 250000, 0.9,name="learning_rate")
        optimizer=tf.train.AdagradOptimizer(self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_step = optimizer.apply_gradients(zip(gradients, variables), name="train_step")
        
        self.reset_optimizer_op = tf.variables_initializer(optimizer.variables(), name="reset_optimizer")

    # def step(self):
    #     gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
    #     gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    #     self.optimizer.apply_gradients(zip(gradients, variables))

    def create_net(self, add_rnn):
        if add_rnn == False:
            self.layers=[self.x_image]

        else:
            LSTM_out = LSTM_layer(self.x).output()
            self.layers = [LSTM_out]
            rnn_output_image = tf.reshape(self.layers[-1], [-1, 1, self.traj_length, 1])
            self.layers.append(rnn_output_image)

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

    def create_loss(self, with_rnn):
        if with_rnn == False:
            recon_loss = math_ops.squared_difference(self.x,self.decoded, name = "recon_loss")
            self.loss = tf.reduce_mean(recon_loss, 0, name = "loss")
        else:
            recon_loss = math_ops.squared_difference(self.true_x,self.decoded, name = "recon_loss")
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


    # Added Network Control tools