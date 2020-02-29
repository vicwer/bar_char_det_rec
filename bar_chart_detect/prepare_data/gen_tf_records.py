# encoding: utf-8

import numpy as np
import tensorflow as tf
import os
import cv2
import sys
sys.path.append('..')
from config import cfg

def load_file(file_path):
    '''
    load imgs_path, classes and labels
    '''
    imgs_path = []
    classes = []
    labels = []
    labels_and_classes = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_path = line.strip().split(' ')[0]
            cls = int(line.strip().split(' ')[1])
            label = [float(i) for i in line.strip().split(' ')[2:]]
            imgs_path.append(img_path)
            classes.append(cls)
            labels.append(label)
            label.append(cls)
            labels_and_classes.append(label)
    return np.asarray(imgs_path), np.asarray(classes), np.asarray(labels), np.array(labels_and_classes)

def extract_image(image_path, height, width, is_resize=True):
    '''
    get r->g->b image data
    '''
    image = cv2.imread(image_path)
    if is_resize:
        image = cv2.resize(image, (width, height))
    image_data = np.array(image, dtype='float32') / 255
    #b, g, r = cv2.split(image)
    #rgb_image = cv2.merge([r, g, b])
    return image

def encode_tfrecords(tf_records_filename, imgs, classes, labels, labels_and_classes):
    '''
    encode img, class and label to tfrecords
    '''
    writer = tf.python_io.TFRecordWriter(tf_records_filename)
    for i in range(labels.shape[0]):
        img = imgs[i].tostring()
        label_and_class = labels_and_classes[i].flatten().tolist()
        example = tf.train.Example(features=tf.train.Features(feature={
                      #'cls': tf.train.Feature(int64_list = tf.train.Int64List(value=[classes[i]])),
                      #'label': tf.train.Feature(float_list = tf.train.FloatList(value=labels[i])),
                      'label_and_class' : tf.train.Feature(float_list = tf.train.FloatList(value=label_and_class)),
                      'feature': tf.train.Feature(bytes_list = tf.train.BytesList(value=[img]))
                  }))
        writer.write(example.SerializeToString())
    writer.close()

def encode_tfrecords_mult_labels(tf_records_filename, imgs, labels):
    '''
    encode img and labels to tfrecords; one image --> mult labels
    '''
    writer = tf.python_io.TFRecordWriter(tf_records_filename)
    for i in range(labels.shape[0]):
        img = imgs[i].tostring()
        cls_num = np.asarray(labels[i]).shape[0] / 8
        print('cls_num', cls_num)
        cls = np.asarray(range(0, int(cls_num))).tostring()
        label = np.asarray(labels[i]).tostring()
        print(np.asarray(range(0, int(cls_num))).shape, np.asarray(labels[i]).shape)
        height = np.array([1080])
        width = np.array([1920])
        cls_num = np.array([int(cls_num)])
        example = tf.train.Example(features=tf.train.Features(feature={
                      'height': tf.train.Feature(int64_list = tf.train.Int64List(value=height)),
                      'width': tf.train.Feature(int64_list = tf.train.Int64List(value=width)),
                      'cls': tf.train.Feature(bytes_list = tf.train.BytesList(value=[cls])),
                      'cls_num': tf.train.Feature(int64_list = tf.train.Int64List(value=cls_num)),
                      'label': tf.train.Feature(bytes_list = tf.train.BytesList(value=[label])),
                      'feature': tf.train.Feature(bytes_list = tf.train.BytesList(value=[img]))
                  }))
        writer.write(example.SerializeToString())
    writer.close()

def run_encode(file_path, tf_records_filename, is_mult_labels=True):
    '''
    encode func
    '''
    imgs_path, classes, labels, labels_and_classes = load_file(file_path)
    height, width = 1080, 1920
    imgs = []
    for i in range(imgs_path.shape[0]):
        img = extract_image(imgs_path[i], height, width, is_resize=False)
        imgs.append(img)
    imgs = np.asarray(imgs)
    if is_mult_labels:
        encode_tfrecords_mult_labels(tf_records_filename, imgs, labels)
    else:
        encode_tfrecords(tf_records_filename, imgs, classes, labels, labels_and_classes)

if __name__ == '__main__':
    file_path = '../data/train_list/train.txt'
    tf_records_filename = cfg.data_path
    run_encode(file_path, tf_records_filename, is_mult_labels=False)
