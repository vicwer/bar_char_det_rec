# encoding: utf-8

import numpy as np
import tensorflow as tf
import os
import cv2
from tqdm import tqdm
import re
import sys
sys.path.append('..')
from config import cfg

def load_file(file_path):
    '''
    load imgs_path, classes and labels
    '''
    imgs_path = []
    labels = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_path = line.strip().split(' ')[0]
            label = [float(i) for i in line.strip().split(' ')[1:]]
            imgs_path.append(img_path)
            labels.append(label)
    return np.asarray(imgs_path), np.asarray(labels)

def extract_image(image_path, height, width, is_resize=True):
    '''
    get b->g->r image data
    '''
    image = cv2.imread(image_path)
    if is_resize:
        image = cv2.resize(image, (width, height))
    #image_data = np.array(image, dtype='float32') / 255.0
    image_data = np.array(image, dtype='uint8')
    return image_data

def run_encode(file_path, tf_records_filename):
    '''
    encode func
    '''
    imgs_path, labels = load_file(file_path)
    height, width = 20, 60
    imgs = []
    writer = tf.python_io.TFRecordWriter(tf_records_filename)
    for i in tqdm(range(imgs_path.shape[0])):
        img = extract_image(imgs_path[i], height, width, is_resize=True)
        img = img.tostring()
        label = labels[i].flatten().tolist()
        example = tf.train.Example(features=tf.train.Features(feature={
                      'label_and_class' : tf.train.Feature(float_list = tf.train.FloatList(value=label)),
                      'feature': tf.train.Feature(bytes_list = tf.train.BytesList(value=[img]))
                  }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    file_path = re.sub(r'prepare_data', '', os.getcwd()) + 'data/train_list/train_list.txt'
    tf_records_filename = cfg.data_path
    #file_path = '/home/xjyu/kgduan/comedy/TF/examples/tf_data/img_and_label_list.txt'
    #tf_records_filename = '/home/xjyu/kgduan/comedy/TF/examples/tf_data/train.records'
    run_encode(file_path, tf_records_filename)
