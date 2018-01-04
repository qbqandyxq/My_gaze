import os

import tensorflow as tf
from datasets import dataset_utils

slim = tf.contrib.slim

def get_split(split_name, dataset_dir, file_pattern, reader,
              split_to_sizes, items_to_descriptions):

    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/labels/label1': tf.VarLenFeature(dtype=tf.float32),
        'image/labels/label2': tf.VarLenFeature(dtype=tf.float32),
        'image/labels/label3': tf.VarLenFeature(dtype=tf.float32),
        'image/labels/label4': tf.VarLenFeature(dtype=tf.float32),
        'image/labels/label5': tf.VarLenFeature(dtype=tf.float32),
        'image/labels/label6': tf.VarLenFeature(dtype=tf.float32),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'images/label1': slim.tfexample_decoder.Tensor('image/labels/label1'),
        'images/label2': slim.tfexample_decoder.Tensor('image/labels/label2'),
        'images/label3': slim.tfexample_decoder.Tensor('image/labels/label3'),
        'images/label4': slim.tfexample_decoder.Tensor('image/labels/label4'),
        'images/label5': slim.tfexample_decoder.Tensor('image/labels/label5'),
        'images/label6': slim.tfexample_decoder.Tensor('image/labels/label6'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    #labels_to_names = None
    #if dataset_utils.has_labels(dataset_dir):
    #    labels_to_names = dataset_utils.read_label_file(dataset_dir)
    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=split_to_sizes[split_name],
            items_to_descriptions=items_to_descriptions,)
