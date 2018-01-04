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
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import tensorflow as tf
from datasets import gaze_common

slim = tf.contrib.slim

FILE_PATTERN = 'voc_%s_*.tfrecord'

SPLITS_TO_SIZES = {
    'train': 37332,
}

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'image/label': 'A list of labels, one per each object.',
}
# (Images, Objects) statistics on every class.

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    if not file_pattern:
        file_pattern = FILE_PATTERN
    return gaze_common.get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,)
