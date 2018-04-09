from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

#from nets import alexnet
from nets import gazebase

#from nets.nasnet import nasnet

slim = tf.contrib.slim

networks_map = {'gazebase': gazebase.alexnet_v2,
               }

arg_scopes_map = {'gazebase': gazebase.alexnet_v2_arg_scope,
                 }


def get_network_fn(name, weight_decay=0.0, is_training=False):
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]
    @functools.wraps(func)
    def network_fn(images, **kwargs):
        arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            return func(images, is_training=is_training, **kwargs)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn