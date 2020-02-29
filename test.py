#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
from bar_chart_detect.models.run_net import BarDetNet
from bar_chart_detect.config import cfg

from number_recognize.models.run_net import NumRecNet
from number_recognize.config import cfg as rec_cfg
import cv2
import os
import re

def detect(img_path, use_gpu):
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    dim_w = 608
    dim_h = 608
    g_step = 60000
    is_training = False
    t = 0.5
    count = 0

    # detect
    imgs_holder = tf.placeholder(tf.float32, shape=[1, dim_h, dim_w, 3])
    model = BarDetNet(imgs_holder, None, is_training)
    img_hw = tf.placeholder(dtype=tf.float32, shape=[2])
    boxes, scores, classes = model.predict(img_hw, iou_threshold=0.5, score_threshold=t)

    saver = tf.train.Saver()
    det_ckpt_dir = cfg.ckpt_path_608
    print(det_ckpt_dir)
    imgs = os.listdir(img_path)

    with tf.Session() as sess:
        if use_gpu:
            configer = tf.ConfigProto()
            configer.gpu_options.per_process_gpu_memory_fraction = 0.3
            sess=tf.Session(config=configer)
        else:
            sess = tf.Session()

        det_ckpt = tf.train.get_checkpoint_state(det_ckpt_dir)
        #saver.restore(sess, ckpt.model_checkpoint_path)
        saver.restore(sess, det_ckpt_dir+str(g_step)+'_charbar.ckpt-'+str(g_step+1))

        with tf.Graph().as_default():
            rec_dim_w = 60
            rec_dim_h = 20
            rec_g_step = 10000
            is_training = False

            rec_imgs_holder = tf.placeholder(tf.float32, shape=[1, rec_dim_h, rec_dim_w, 3])
            rec_model = NumRecNet(rec_imgs_holder, None, is_training)
            score_index = rec_model.predict()

            rec_saver = tf.train.Saver()
            rec_ckpt_dir = rec_cfg.ckpt_path_608

            with tf.Session() as sess2:
                if use_gpu:
                    configer = tf.ConfigProto()
                    configer.gpu_options.per_process_gpu_memory_fraction = 0.3
                    sess2 = tf.Session(config=configer)
                else:
                    sess2 = tf.Session()

                rec_ckpt = tf.train.get_checkpoint_state(rec_ckpt_dir)
                rec_saver.restore(sess2, rec_ckpt_dir+str(rec_g_step)+'_coord.ckpt-'+str(rec_g_step+1))

        for i in imgs:
            if 'png' not in i:
                continue

            image_path = os.path.join(img_path, i)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (608,608))
            #cv2.imshow('original img', image)
            #cv2.waitKey(0)
            h, w, c = image.shape
            image_data = np.array(image, dtype='float32') / 255.0

            boxes_, scores_, classes_ = sess.run([boxes, scores, classes], feed_dict={img_hw:[h ,w], imgs_holder: np.reshape(image_data, [1, dim_h, dim_w, 3])})

            boxes_ = boxes_[:, [1, 0, 3, 2]]
            coord_index = classes_ == 1
            coord_boxes = boxes_[coord_index].astype(np.int64)

            coord_list = []
            for i in range(coord_boxes.shape[0]):
                coord_box = coord_boxes[i]
                coord_img = image[coord_box[1]:coord_box[3], coord_box[0]:coord_box[2], :]
                coord_img = cv2.resize(coord_img, (60, 20))
                coord_img = np.array(coord_img, dtype='float32') / 128.0
                scores_index_ = sess2.run([score_index], feed_dict={rec_imgs_holder: np.reshape(coord_img, [1, rec_dim_h, rec_dim_w, 3])})
                label = np.array(rec_cfg.names)[scores_index_][0]
                coord = ''
                for i in range(len(label)):
                    if '$' == label[i]:
                        break
                    else:
                        coord += label[i]
                coord_list.append(coord)

            img = np.floor(image_data * 255 + 0.5).astype('uint8')
            for i in range(boxes_.shape[0]):
                box = boxes_[i]
                x_left, y_top, x_right, y_bottom= box
                cv2.rectangle(img, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (0,0,0), 3)
                if coord_index[i] == True:
                    coord = coord_list.pop(0)
                    cv2.putText(img, coord, (int(x_left), int(y_top)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                
            #cv2.imshow('res', img)
            cv2.imwrite("./detRecRes/res"+str(count)+".png", img)
            count += 1
            #cv2.waitKey(0)

if __name__ == '__main__':
    image_path = './bar_chart_detect/data'
    use_gpu = True
    detect(image_path, use_gpu)
