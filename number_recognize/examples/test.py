#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
from models.run_net import NumRecNet
from prepare_data.gen_data_batch import gen_data_batch
from config import cfg
import cv2
import os
import re
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def resize_image(image_path, w, h, need_resize):
    image = cv2.imread(image_path)
    if need_resize:
        image = cv2.resize(image, (60, 20))
    return image


def test(img_path):
    dim_w = 20
    dim_h = 60
    g_step = 60000

    need_resize = True if dim_w%20==0 and dim_h%60==0 else False
    is_training = False
    cfg.batch_size = 1

    imgs_holder = tf.placeholder(tf.float32, shape=[1, dim_h, dim_w, 3])
    model = NumRecNet(imgs_holder, None, is_training)
    score_index = model.predict()

    saver = tf.train.Saver()
    ckpt_dir = re.sub(r'examples/', '', cfg.ckpt_path_608)

    imgs = os.listdir(img_path)

    with tf.Session() as sess:
        configer = tf.ConfigProto()
        configer.gpu_options.per_process_gpu_memory_fraction = 0.3
        sess=tf.Session(config=configer)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        print(ckpt.model_checkpoint_path)
        #saver.restore(sess, ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt_dir+str(g_step)+'_plate.ckpt-'+str(g_step+1))

        for i in imgs:
            if 'png' not in i:
                continue

            image_path = os.path.join(img_path, i)

            image = resize_image(image_path, dim_w, dim_h, need_resize)
            h, w, c = image.shape
            print(h, w)
            image_data = np.array(image, dtype='float32') / 255.0

            scores_index_ = sess.run([score_index], feed_dict={imgs_holder: np.reshape(image_data, [1, dim_h, dim_w, 3])})

            print(scores_index_)
            cv2.imshow('res', image)
            cv2.waitKey(0)

if __name__ == '__main__':
    image_path = '/diskdt/dataset/coord_dataset/coord_batch_1'
    test(image_path)
