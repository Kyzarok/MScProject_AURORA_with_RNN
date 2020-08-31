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


def LSTM_layer(input):
    #10^-3
    key = 0.001
    mult = int(1/key)
    n_input = 50

    # dictionary = dict()
    # print("Begin Creating Dictionaries")
    # x_val = tf.constant(0, dtype = tf.float32)
    # total = tf.constant(0, dtype = tf.float32)

    # # x = tf.constant([5, 4, 6, 7])
    # # y = tf.constant([5, 2, 5, 10])
    # # print(tf.math.greater_equal(x, y))
    

    # # print(tf.math.greater_equal(x_val, total))
    # for i in range(mult + 1):
    #     y_val = tf.constant(0, dtype = tf.float32)
    #     dictionary[x_val] = dict()
    #     for j in range(mult + 1):
    #         dictionary[x_val][y_val] = total
    #         total += 1
    #         y_val += key
    #     x_val += key
    #     print(x_val)

    
    # reverse_dictionary = dict()

    # for k, v in dictionary.items():
    #     print(k)
    #     for subk, subv in v.items():
    #         reverse_dictionary[subv] = [k, subk]

    # print("Finished Creating Dictionaries")

    # Consider what the max possible size will be. IFwe can round the input values in to the nearest 0.001
    # we can say that for each dimension (1 - 0) / 0.001 = 1000

    # multiplier = tf.constant(mult, dtype = tf.float32)
    # tf.round(x * multiplier) / multiplier

    # # Calculated encoded version of time series of input data
    # grid_encoded_input = []
    # a_mod = (mult * mult) + 1
    # b_mod = mult
    # for i in range(n_input):
    #     dim_0 = x[0][i]
    #     dim_1 = x[0][i + n_input]
    #     a = tf.round(dim_0 * multiplier) / multiplier
    #     b = tf.round(dim_1 * multiplier) / multiplier
    #     # a = a * a_mod
    #     # b = b * b_mod
    #     index = dictionary[a][b]
    #     grid_encoded_input.append(index)

    # grid_encoded_input = tf.convert_to_tensor(grid_encoded_input, dtype = tf.float32)
    # print("State of encoded:")
    # print(grid_encoded_input)
     
    # vocab_size = len(dictionary)
    vocab_size = ( (mult * mult) + mult ) + mult + 1

    # number of units in RNN cell
    n_hidden = 512 # STANDARD VALUE

    # RNN output node weights and biases
    weights = {'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]), trainable=True)}
    biases = {'out': tf.Variable(tf.random_normal([vocab_size]), trainable=True)}

    # reshape to [1, n_input]
    input_x = tf.reshape(input, [-1, n_input])
    # print("Result of reshape")
    # print(input_x)

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    input_x = tf.split(input_x, n_input,1)
    # print(input_x)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    # generate prediction
    # outputs, states = tf.nn.rnn_cell.static_rnn(rnn_cell, x, dtype=tf.float32)
    outputs, _ = tf.nn.static_rnn(rnn_cell, input_x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    true_outputs = tf.matmul(outputs, weights['out']) + biases['out']
    # print(true_outputs.shape)
    # true_outputs = true_outputs.eval(session = 'sess')

    # decoded_output = np.zeros((1, 100))
    # const = tf.constant(1001, dtype = tf.int64)
    # print()
    # for i in range(n_input):
    #     tmp = tf.math.argmax(true_outputs[i][0])
    #     a, b = reverse_dictionary[tmp]
    #     decoded_output[0][i] = a 
    #     decoded_output[0][i + n_input] = b

    # for i in range(n_input):
    #     tmp = tf.math.argmax(true_outputs[i][0])
        
    #     a = 0
    #     while tf.greater_equal(tmp, const)[0]:
    #         tmp -= 1001
    #         a += 1
    #     decoded_output[0][i] = a / a_mod
    #     decoded_output[0][i + n_input] = tf.float32(tmp) / b_mod
    print("lstm")
    print(true_outputs.shape)
    return true_outputs
