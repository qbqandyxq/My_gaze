from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
from collections import namedtuple
import math
import numpy as np
GazeParams = namedtuple('GazeParameters', ['img_shape','labels'])


class GazeNet(object):
    default_params = GazeParams(
        img_shape=(448, 448),
        labels=6,)

    def __init__(self, params=None):
        if isinstance(params, GazeParams):
            self.params = params
        else:
            self.params = GazeNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            reuse=None,
            scope='gaze_alexnet'):
        r = alexnet_v2(inputs,
                    labels=self.params.labels,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    reuse=reuse,
                    scope=scope)
        return r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        return alexnet_v2_arg_scope(weight_decay, data_format=data_format)

    def losses(self, logits, glabels, scope='gaze_losses'):
        return gaze_losses(logits, glabels, scope=scope)


def alexnet_v2_arg_scope(weight_decay=0.0005,data_format='NHWC'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      biases_initializer=tf.constant_initializer(0.1),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc


def alexnet_v2(inputs,
               labels=6,
               is_training=True,
               dropout_keep_prob=0.5,
               scope='gaze_alexnet',
               reuse=None,
               global_pool=False):
 
  with tf.variable_scope(scope, 'gaze_alexnet', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],):
                        #outputs_collections=[end_points_collection]):
        net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
        # 110,110,64
        net = slim.max_pool2d(net, [3, 3], 2, padding="SAME",scope='pool1')
        # 55,55,64
        net = slim.conv2d(net, 192, [5, 5], scope='conv2')
        # 55,55,192
        net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
        # 27,27,192
        net = slim.conv2d(net, 384, [3, 3], scope='conv3')
        # 27,27,384
        net = slim.conv2d(net, 384, [3, 3], scope='conv4')
        # 27,27,384
        net = slim.conv2d(net, 256, [3, 3], scope='conv5')
        # 27,27,256
        net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
        # 13, 13, 256

      # Use conv2d instead of fully_connected layers.

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          weights_initializer=trunc_normal(0.005),
                          biases_initializer=tf.constant_initializer(0.1)):
        
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        net_ = slim.conv2d(net, 256, [1, 1], scope='convs_1')
        net_ = slim.conv2d(net_, 256, [1, 1], scope='convs_2')
        net_ = slim.conv2d(net_, 1, [1, 1], scope='convs_3')

        #change the dimension
        #net_ = tf.stack
        net = tf.multiply(net , tf.concat([net_]*256,3) )#
        print("==========", net.shape)
        net = tf.reshape(net, [16,-1])
        net = slim.fully_connected(net, 4096, scope='fc6')

        net = slim.dropout(net, 0.5, scope='dropout6')  
        net = slim.fully_connected(net, 4096, scope='fc7') 
        net = slim.dropout(net, 0.5, scope='dropout7')  
        
        net = slim.fully_connected(net, 6, activation_fn=None, scope='fc8') 
        end_points['final8_layer'] = net
        return net #, end_points
    
    
def gaze_losses(logits, glabels, scope=None):
    with tf.name_scope('loss'):
        loss = tf.abs(logits - glabels)
        cost = tf.reduce_mean(loss)
        with tf.name_scope('loss'):
            tf.losses.add_loss(cost)
            

def gaze_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format) as sc:
            return sc
    
alexnet_v2.default_image_size = (448, 448)