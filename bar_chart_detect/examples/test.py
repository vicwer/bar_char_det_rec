#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
from models.run_net import BarDetNet
from prepare_data.gen_data_batch import gen_data_batch
from config import cfg
import cv2
import os
import re
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def test(img_path):
    dim_w = 608
    dim_h = 608
    g_step = 40000
    is_training = False
    t = 0.5

    imgs_holder = tf.placeholder(tf.float32, shape=[1, dim_h, dim_w, 3])
    model = BarDetNet(imgs_holder, None, is_training)
    img_hw = tf.placeholder(dtype=tf.float32, shape=[2])
    boxes, scores, classes = model.predict(img_hw, iou_threshold=0.5, score_threshold=t)

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
        saver.restore(sess, ckpt_dir+str(g_step)+'_charbar.ckpt-'+str(g_step+1))

        for i in imgs:
            if 'png' not in i:
                continue

            image_path = os.path.join(img_path, i)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (608,608))
            h, w, c = image.shape
            image_data = np.array(image, dtype='float32') / 255.0

            boxes_, scores_, classes_ = sess.run([boxes, scores, classes], feed_dict={img_hw:[h ,w], imgs_holder: np.reshape(image_data, [1, dim_h, dim_w, 3])})

            print(scores_)
            img = np.floor(image_data * 255 + 0.5).astype('uint8')
            for i in range(boxes_.shape[0]):
                box = boxes_[i]
                y_top, x_left, y_bottom, x_right = box
                cv2.rectangle(img, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (0,0,0), 1)
            cv2.imshow('res', img)
            cv2.waitKey(0)

if __name__ == '__main__':
    image_path = '../data/test_imgs/'
    test(image_path)
