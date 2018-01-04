import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image, ImageChops

def rm_black(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    
    
def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def read_text(dir_path, ori_path, output_dir, name, modify_path, person, diff_fidx):
    with open(dir_path, 'rt') as f:
        data = f.readlines()
        

    SAMPLES_PER_FILES=200
    i = 0
    fidx = diff_fidx
    while i < len(data):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(data) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(data)))
                sys.stdout.flush()

                one_line = data[i].split(' ')
                img_path= os.path.join(ori_path, one_line[0])
                
                
                '''
                im=Image.open(img_path)
                im_crop = rm_black(im)
                modify_to_person = os.path.join(modify_path, person)
                modify_to_day = os.path.join(modify_to_person, one_line[0][:5])
                if not os.path.exists(modify_to_day):
                    os.mkdir(modify_to_day)
                modify_path_end = os.path.join(modify_to_person, one_line[0])
                im_crop.save(modify_path_end)
                '''
                
                #image_data = tf.gfile.FastGFile(modify_path_end, 'rb').read()
                image_data = tf.gfile.FastGFile(img_path, 'rb').read()
                labels = one_line[21:27]
                labels = [float(i) for i in labels]
                examlpe = _convert_to_example(image_data, labels)
                tfrecord_writer.write(examlpe.SerializeToString())
                #print("write into tfrecord :", fidx)
                #_add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1
    return fidx

def _convert_to_example(image_data, labels):

    image_format = b'jpg'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/labels/label1': float_feature(labels[0]),
            'image/labels/label2': float_feature(labels[1]),
            'image/labels/label3': float_feature(labels[2]),
            'image/labels/label4': float_feature(labels[3]),
            'image/labels/label5': float_feature(labels[4]),
            'image/labels/label6': float_feature(labels[5]),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example
    
#def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
#    example = _convert_to_example(image_data, labels)
#    tfrecord_writer.write(example.SerializeToString())
    
    
def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)
        
def main(_):
    dir_path='/nfshome/xueqin/udalearn/gaze_estimation/data/MPIIFaceGaze/'
    modify_path='/nfshome/xueqin/udalearn/gaze_estimation/data/modify_data/'
    
    #filenames = sorted(os.listdir(modify_path))[:15]
    filenames = sorted(os.listdir(dir_path))[:15]
    output_dir='/nfshome/xueqin/udalearn/gaze_estimation/data/tf_data'
    name='voc_train'
    
    
    i = 0
    diff_fidx = 0
    for person in filenames:
        #while i < len(filenames):

        person_modify_path=os.path.join(modify_path, person)
        print("\nwriting into :", person_modify_path)
        print("==========================================")
                #mk files for every person
        if not os.path.exists(person_modify_path):
            os.mkdir(person_modify_path)
                    
        print('The rest file %d/%d' % (i+1, len(filenames)))
                
        path = os.path.join(dir_path, person)
                # person path
        txt_data = sorted(os.listdir(path))[-1]

        txt_path = os.path.join(path, txt_data) 
        diff_fidx = read_text(txt_path, path, output_dir, name, modify_path, person, diff_fidx)
        i += 1
        diff_fidx +=1
    print('\nFinished converting the dataset!')
    
if __name__ == '__main__':
    tf.app.run()