from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def alexnet_v2_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      biases_initializer=tf.constant_initializer(0.1),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc


def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2',
               global_pool=False):
  """AlexNet version 2.
  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224 or set
        global_pool=True. To use in fully convolutional mode, set
        spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: the number of predicted classes. If 0 or None, the logits layer
    is omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      logits. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original AlexNet.)
  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0
      or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
        net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                        scope='conv1')
        # 110,110,64
        net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
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

    with slim.arg_scope([slim.conv2d],
                          weights_initializer=trunc_normal(0.005),
                          biases_initializer=tf.constant_initializer(0.1)):
        net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                          scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)
        if global_pool:
            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
            end_points['global_pool'] = net
        if num_classes:
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                             scope='dropout7')
            net = slim.conv2d(net, num_classes, [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            biases_initializer=tf.zeros_initializer(),
                            scope='fc8')
            if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
            end_points[sc.name + '/fc8'] = net
        return net, end_points
alexnet_v2.default_image_size = 448