# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pre-processing images for SSD-type networks.
"""
from enum import Enum, IntEnum
import numpy as np

import tensorflow as tf


from tensorflow.python.ops import control_flow_ops

from preprocessing import tf_image
#from nets import ssd_common

slim = tf.contrib.slim

# Resizing strategies.
Resize = IntEnum('Resize', ('NONE',                # Nothing!
                            'CENTRAL_CROP',        # Crop (and pad if necessary).
                            'PAD_AND_RESIZE',      # Pad, and resize to output shape.
                            'WARP_RESIZE'))        # Warp resize.

# VGG mean parameters.
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

# Some training pre-processing parameters.
BBOX_CROP_OVERLAP = 0.5         # Minimum overlap to keep a bbox after cropping.
MIN_OBJECT_COVERED = 0.25
CROP_RATIO_RANGE = (0.6, 1.67)  # Distortion ratio during cropping.
EVAL_SIZE = (448, 448)


def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image


def tf_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary.

    Returns:
      Centered image.
    """
    mean = tf.constant(means, dtype=image.dtype)
    image = image + mean
    if to_int:
        image = tf.cast(image, tf.int32)
    return image


def np_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary. Numpy version.

    Returns:
      Centered image.
    """
    img = np.copy(image)
    img += np.array(means, dtype=img.dtype)
    if to_int:
        img = img.astype(np.uint8)
    return img


def tf_summary_image(image, bboxes, name='image', unwhitened=False):
    """Add image with bounding boxes to summary.
    """
    if unwhitened:
        image = tf_image_unwhitened(image)
    image = tf.expand_dims(image, 0)
    bboxes = tf.expand_dims(bboxes, 0)
    image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
    tf.summary.image(name, image_with_box)

    
def preprocess_for_train(image,
                         out_shape, data_format='NHWC',
                         scope='gaze_preprocessing_train'):

    fast_mode = False
    with tf.name_scope(scope, 'gaze_preprocessing_train', [image]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Resize image to output size.
        resize_image = tf_image.resize_image(image, out_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
        # Rescale to VGG input scale.
        image = resize_image * 255.
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, 


def preprocess_for_eval(image,
                        out_shape=EVAL_SIZE, data_format='NHWC',
                        difficults=None, resize=Resize.WARP_RESIZE,
                        scope='ssd_preprocessing_train'):
    """Preprocess an image for evaluation.

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        out_shape: Output shape after pre-processing (if resize != None)
        resize: Resize strategy.

    Returns:
        A preprocessed image.
    """
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        image = tf.to_float(image)
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])

        # Add image rectangle to bboxes.
        bbox_img = tf.constant([[0., 0., 1., 1.]])
        if bboxes is None:
            bboxes = bbox_img
        else:
            bboxes = tf.concat([bbox_img, bboxes], axis=0)

        if resize == Resize.NONE:
            # No resizing...
            pass
        elif resize == Resize.CENTRAL_CROP:
            # Central cropping of the image.
            image, bboxes = tf_image.resize_image_bboxes_with_crop_or_pad(
                image, bboxes, out_shape[0], out_shape[1])
        elif resize == Resize.PAD_AND_RESIZE:
            # Resize image first: find the correct factor...
            shape = tf.shape(image)
            factor = tf.minimum(tf.to_double(1.0),
                                tf.minimum(tf.to_double(out_shape[0] / shape[0]),
                                           tf.to_double(out_shape[1] / shape[1])))
            resize_shape = factor * tf.to_double(shape[0:2])
            resize_shape = tf.cast(tf.floor(resize_shape), tf.int32)

            image = tf_image.resize_image(image, resize_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
            # Pad to expected size.
            image, bboxes = tf_image.resize_image_bboxes_with_crop_or_pad(
                image, bboxes, out_shape[0], out_shape[1])
        elif resize == Resize.WARP_RESIZE:
            # Warp resize of the image.
            image = tf_image.resize_image(image, out_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)

        # Split back bounding boxes.
        bbox_img = bboxes[0]
        bboxes = bboxes[1:]
        # Remove difficult boxes.
        if difficults is not None:
            mask = tf.logical_not(tf.cast(difficults, tf.bool))
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)
        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes, bbox_img


def preprocess_image(image,
                     out_shape,
                     data_format,
                     is_training=False,
                     **kwargs):
    """Pre-process an given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, then this value
        is used for rescaling.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, this value is
         ignored. Otherwise, the resize side is sampled from
         [resize_size_min, resize_size_max].

    Returns:
      A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image,
                                    out_shape=out_shape,
                                    data_format=data_format)
    else:
        return preprocess_for_eval(image, 
                                   out_shape=out_shape,
                                   data_format=data_format,
                                   **kwargs)
