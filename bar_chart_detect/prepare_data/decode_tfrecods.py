# encoding : uft-8

import numpy as np
import tensorflow as tf
import os
import cv2
from scipy import misc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def decode_tfrecords(tf_records_filename, is_batch=False, is_shuffle=True):
    '''
    load image and label from tf records
    '''
    input_queue = tf.train.string_input_producer([tf_records_filename], num_epochs=2, shuffle=False)
    reader = tf.TFRecordReader()
    key, value = reader.read(input_queue)

    features = tf.parse_single_example(value, features={#'cls': tf.FixedLenFeature([1], tf.int64),
                                                        #'label': tf.FixedLenFeature([4], tf.float32),
                                                        'label_and_class' : tf.FixedLenFeature([5], tf.float32),
                                                        'feature': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['feature'], tf.float32)
    image = tf.reshape(image, [1080, 1920, 3])
    #label = tf.cast(features['label'], tf.float32)
    #label = features['label']
    #cls = features['cls']
    label_and_class = features['label_and_class']
    label_and_class = tf.reshape(label_and_class, [1, 5])

    if is_batch:
        batch_size = 2
        min_after_dequeue = 10
        num_threads = 2
        capacity = min_after_dequeue + batch_size * num_threads
        if is_shuffle:
            #image, label, cls = tf.train.shuffle_batch([image, label, cls], batch_size=batch_size, num_threads=num_threads, capacity=capacity, min_after_dequeue=min_after_dequeue)
            image, label_and_class = tf.train.shuffle_batch([image, label_and_class], batch_size=batch_size, num_threads=num_threads, capacity=capacity, min_after_dequeue=min_after_dequeue)
        else:
            image, label_and_class = tf.train.batch([image, label_and_class], batch_size=batch_size, num_threads=num_threads, capacity=capacity)

    with tf.Session() as sess:
        configer = tf.ConfigProto()
        configer.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess=tf.Session(config=configer)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #init_op = tf.global_variables_initializer()
        #sess.run(init_op)
        with tf.Graph().as_default(), tf.device('/device:CPU:0'):
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop():
                    image_batch, label_and_class_batch = sess.run([image, label_and_class])
                    print(label_and_class_batch)
            except tf.errors.OutOfRangeError:
                print('done')
            finally:
                coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf_records_filename = '../data/tf_data/train.records'
    decode_tfrecords(tf_records_filename, is_batch=True, is_shuffle=False)
