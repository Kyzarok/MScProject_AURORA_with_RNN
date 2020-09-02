#
#   my_nn_lib.py
#       date. 5/19/2016
#

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import numpy as np
# import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import tensorflow as tf2

def xavier_init(shape, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(1.0/np.sum(shape))
    high = constant*np.sqrt(1.0/np.sum(shape))
    return tf.random_uniform(shape,  minval=low, maxval=high,  dtype=tf.float32)
#stddev = np.sqrt(1.0 / (fan_in + fan_out))
#return tf.random_normal((fan_in, fan_out), mean = 0.0, stddev=stddev, dtype=tf.float64)




# Convolution 2-D Layer
class Convolution2D(object):
    '''
      constructor's args:
          input     : input image (2D matrix)
          input_siz ; input image size
          in_ch     : number of incoming image channel
          out_ch    : number of outgoing image channel
          patch_siz : filter(patch) size
          weights   : (if input) (weights, bias)
    '''
    def __init__(self, input, input_siz, in_ch, out_ch, patch_siz, activation='relu'):
        self.input = input      
        self.rows = input_siz[0]
        self.cols = input_siz[1]
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.activation = activation

        wshape = [patch_siz[0], patch_siz[1], in_ch, out_ch]
        
        #w_cv = tf.Variable(tf.truncated_normal(wshape, stddev=0.1), 
        #                    trainable=True)
        #b_cv = tf.Variable(tf.constant(0.1, shape=[out_ch]), 
        #                    trainable=True)

        w_cv = tf.Variable(xavier_init(wshape), trainable=True)
        b_cv = tf.Variable(tf.zeros(shape=[out_ch]), trainable=True)
        
        self.w = w_cv
        self.b = b_cv
        self.params = [self.w, self.b]
        
    def output(self):
        shape4D = [-1, self.rows, self.cols, self.in_ch]
        
        x_image = tf.reshape(self.input, shape4D)  # reshape to 4D tensor
        linout = tf.nn.conv2d(x_image, self.w, 
                  strides=[1, 1, 1, 1], padding='SAME') + self.b
        shapeOut = [None, self.rows, self.cols, self.out_ch]
        linout.set_shape(shapeOut)
        print("conv: ")
        print(shapeOut)                 
        if self.activation == 'relu':
            self.output = tf.nn.relu(linout)
        elif self.activation == 'sigmoid':
            self.output = tf.sigmoid(linout)
        elif self.activation == 'truncated_linear':
            self.output = tf.clip_by_value(linout,-1, 1)
        elif self.activation == 'leaky_relu':
            self.output = tf.nn.leaky_relu(linout)
        else:
            self.output = linout
        
        return self.output

# Max Pooling Layer   
class MaxPooling2D(object):
    '''
      constructor's args:
          input  : input image (2D matrix)
          ksize  : pooling patch size
    '''
    def __init__(self, input, ksize=None):
        self.input = input
        if ksize == None:
            ksize = [1, 2, 2, 1]
            self.ksize = ksize
    
    def output(self):
        self.output = tf.nn.max_pool(self.input, ksize=self.ksize,
                    strides=[1, 1, 1, 1], padding='SAME')
        print("max pool")
        print(self.output.get_shape())
        return self.output


class ScaleLayer(object):
    def __init__(self, n_in):
        diagonal = tf.Variable(tf.ones([1,n_in]), trainable=False)
        self.w_in = tf.squeeze(tf.matrix_diag(diagonal))
        self.w_out = tf.squeeze(tf.matrix_diag(tf.reciprocal(diagonal)))
        
    def scale_in(self, input, name=''):
        return tf.matmul(input, self.w_in, name=name)

    def scale_out(self, input, name=''):
        return tf.matmul(input, self.w_out, name=name)
        
# Full-connected Layer   
class FullConnected(object):
    def __init__(self, input, n_in, n_out,activation='relu', name=''):
        self.input = input
    
        #w_h = tf.Variable(tf.truncated_normal([n_in,n_out],
        #                  mean=0.0, stddev=0.05), trainable=True)
        #b_h = tf.Variable(tf.zeros([n_out]), trainable=True)
        
        w_h = tf.Variable(xavier_init([n_in,n_out]), trainable=True)
        b_h = tf.Variable(tf.zeros(shape=[n_out]), trainable=True)

                


        if activation == 'relu':
            print("relu")
            self.activation_fun = tf.nn.relu
        elif activation == 'leaky_relu':
            print("leaky relu")
            self.activation_fun = tf.nn.leaky_relu
        elif activation == 'truncated_linear':
            self.activation_fun = self.trunc_act

        elif activation == 'sigmoid':
            print("sigmoid")
            self.activation_fun = tf.tanh
        else:
            print("identity")
            self.activation_fun = tf.identity

        
        self.w = w_h
        self.b = b_h
        self.params = [self.w, self.b]
        self.name = name

    def trunc_act(self, linout, name =''):
        return tf.clip_by_value(linout,-1, 1,name)
    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        if(self.name!=''):
            self.output = self.activation_fun(linarg,name=self.name)
        else:
            self.output = self.activation_fun(linarg)
        print("FC")
        print(self.output.get_shape())
                        
        return self.output

# Read-out Layer
class ReadOutLayer(object):
    def __init__(self, input, n_in, n_out):
        self.input = input
        
        #w_o = tf.Variable(tf.random_normal([n_in,n_out],
        #                mean=0.0, stddev=0.05), trainable=True)
        #b_o = tf.Variable(tf.zeros([n_out]), trainable=True)
        w_o = tf.Variable(xavier_init([n_in,n_out]), trainable=True)
        b_o = tf.Variable(tf.zeros(shape=[n_ch]), trainable=True)

                 
        
        self.w = w_o
        self.b = b_o
        self.params = [self.w, self.b]
    
    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        self.output = tf.nn.sigmoid(linarg)
        #self.output = linarg

        return self.output
#

# Up-sampling 2-D Layer (deconvolutoinal Layer)
class Conv2Dtranspose(object):
    '''
      constructor's args:
          input      : input image (2D matrix)
          output_siz : output image size
          in_ch      : number of incoming image channel
          out_ch     : number of outgoing image channel
          patch_siz  : filter(patch) size
    '''
    def __init__(self, input, output_siz, in_ch, out_ch, patch_siz, activation='relu'):
        self.input = input      
        self.rows = output_siz[0]
        self.cols = output_siz[1]
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.activation = activation
        
        wshape = [patch_siz[0], patch_siz[1], out_ch, in_ch]    # note the arguments order
        
        #w_cvt = tf.Variable(tf.truncated_normal(wshape, stddev=0.1), 
        #                    trainable=True)
        #b_cvt = tf.Variable(tf.constant(0.1, shape=[out_ch]), 
        #                    trainable=True)
        w_cvt = tf.Variable(xavier_init(wshape), trainable=True)
        b_cvt = tf.Variable(tf.zeros(shape=[out_ch]), trainable=True)
        

        

        self.batsiz = tf.shape(input)[0]
        self.w = w_cvt
        self.b = b_cvt
        self.params = [self.w, self.b]
        
    def output(self):
        shape4D = [self.batsiz, self.rows, self.cols, self.out_ch]

        linout = tf.nn.conv2d_transpose(self.input, self.w, output_shape=shape4D,
                            strides=[1, 1, 1, 1], padding='SAME') + self.b
        
        shapeOut = [None, self.rows, self.cols, self.out_ch]
        print("deconv" )
        print(shapeOut)
        linout.set_shape(shapeOut)
        if self.activation == 'relu':
            self.output = tf.nn.relu(linout)
        elif self.activation == 'leaky_relu':
            self.output = tf.nn.leaky_relu(linout)
        elif self.activation == 'sigmoid':
            self.output = tf.sigmoid(linout)
        else:
            self.output = linout
        
        return self.output

class LSTM_layer(object):
    def __init__(self, input):
        #10^-3
        self.frac = 0.0075
        self.mult = int(2/self.frac)
        self.n_input = 50

        # Max possible size of a_mod * a + b
        self.vocab_size = (self.mult * (self.mult + 1)) + self.mult
        # self.vocab_size = self.mult * self.mult

        # self.vocab_size = ( (self.mult * self.mult) + self.mult ) + self.mult + 1

        # number of units in RNN cell
        self.n_hidden = 128 # STANDARD VALUE

        # RNN output node weights and biases
        self.weights = tf.Variable(xavier_init([self.n_hidden, self.vocab_size]), trainable=True)
        self.biases = tf.Variable(xavier_init([self.vocab_size]), trainable=True)

        # reshape to [1, n_input]
        input_x = tf.reshape(input, [-1, self.n_input])

        # Generate a n_input-element sequence of inputs
        self.input_x = tf.split(input_x, self.n_input,1)

        # 1-layer LSTM with n_hidden units.
        self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)

    def output(self):
        # generate prediction
        outputs, _ = tf.nn.static_rnn(self.rnn_cell, self.input_x, dtype=tf.float32)
        # there are n_input outputs but
        self.true_outputs = tf.matmul(outputs, self.weights) + self.biases

        print("lstm")
        print(self.true_outputs.shape)

        return self.true_outputs
