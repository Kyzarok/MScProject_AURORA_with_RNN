from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.python.ops import math_ops

from original_my_nn_lib import Convolution2D, MaxPooling2D, Conv2Dtranspose
from original_my_nn_lib import FullConnected
from original_my_nn_lib import LSTM_layer


class AE(object):
    lr_init = 0.1
    lr_decay = 0.9
    clip_norm=5.0
    traj_length = 50
    latent_dim = 2
    output_neurons = 100
    dense_neurons = 5
    patch_size_convol=(2, 6)
    patch_size_pool=[1, 2, 2, 1]
    spatial_dims=2
    output_channels=2
    def __init__(self, with_rnn, num_epoch):
        # Same settings for both networks
        self.n_epoch = num_epoch

        self.global_step = tf.placeholder(tf.int32, shape=(), name="step_id")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.x = tf.placeholder(tf.float32, [None, self.traj_length*2], name="input_x")
        self.x_image = tf.reshape(self.x, [-1, self.spatial_dims, self.traj_length, 1])

        if with_rnn == True:
            self.true_x = tf.placeholder(tf.float32, [None, self.traj_length*self.spatial_dims], name="true_x")
            self.new_rnn_input = tf.placeholder(tf.float32, [None, self.traj_length], name="rnn_input")

        self.create_net(with_rnn)

        self.create_loss(with_rnn)
        self.create_optimizer()
        self.saver = tf.train.Saver(tf.trainable_variables())

    def create_optimizer(self):
        self.learning_rate = tf.train.exponential_decay(self.lr_init, self.global_step, self.n_epoch, self.lr_decay,name="learning_rate")
        optimizer=tf.train.AdagradOptimizer(self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        self.train_step = optimizer.apply_gradients(zip(gradients, variables), name="train_step")
        
        self.reset_optimizer_op = tf.variables_initializer(optimizer.variables(), name="reset_optimizer")

    def create_net(self, with_rnn):
        if with_rnn == True:
            self.rnn_output_image = LSTM_layer(self.new_rnn_input).output()
            print(self.rnn_output_image.shape)

        self.layers=[self.x_image]
        with tf.variable_scope('encoder') as vs:
            self.create_encoder_conv([self.output_channels])
            self.create_encoder_fc([self.dense_neurons])

            self.latent = FullConnected(self.layers[-1], self.layers[-1].get_shape().as_list()[1], self.latent_dim, activation='identity', name = "latent").output()
            self.layers.append(self.latent)

        with tf.variable_scope('decoder') as vs:
            self.create_decoder_fc([self.dense_neurons, self.traj_length * self.spatial_dims],[-1, self.traj_length, self.spatial_dims, 1])
            self.create_decoder_conv([self.output_channels,1])
            
            print("Last layer")
            print (self.layers[-1].get_shape().as_list())
            res=tf.reshape(self.layers[-1], [-1, self.traj_length*self.spatial_dims], name = "reconstructed")
            self.layers.append(res)
            self.decoded  = FullConnected(self.layers[-1], self.layers[-1].get_shape().as_list()[1], self.output_neurons, activation = 'identity', name = "decoded").output()

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
            # input, input_siz, in_ch, out_ch, patch_siz, activation='relu'
            conv = Convolution2D(input=self.layers[-1], 
                input_siz=(self.layers[-1].get_shape().as_list()[1], self.layers[-1].get_shape().as_list()[2]), 
                in_ch=self.layers[-1].get_shape().as_list()[3], 
                out_ch=conf[i],
                patch_siz=self.patch_size_convol, 
                activation='leaky_relu')
            self.layers.append(conv.output())
            pool = MaxPooling2D(input=self.layers[-1],ksize=self.patch_size_pool) # 2x2 patch by default
            self.layers.append(pool.output())

    def create_decoder_conv(self, conf):
        for i in range(len(conf)):
            conv_t = Conv2Dtranspose(self.layers[-1],(self.layers[-1].get_shape().as_list()[1], self.layers[-1].get_shape().as_list()[2]), self.layers[-1].get_shape().as_list()[3], conf[i],
                                     self.patch_size_convol, activation='leak_relu')
            self.layers.append(conv_t.output())
