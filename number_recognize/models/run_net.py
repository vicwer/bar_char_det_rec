#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys
sys.path.append('..')
from number_recognize.models.network import Network
from number_recognize.config import cfg
from number_recognize.models.losses import loss

class NumRecNet:
    def __init__(self, img, truth, is_training, batcn_norm_decay=0.997):
        self.img = img
        self.truth = truth
        self.is_training = is_training
        self.batch_norm_decay = batcn_norm_decay
        self.img_shape = tf.shape(self.img)
        backbone = Network()
        if is_training:
            self.head, self.l2_loss = backbone.inference(self.is_training, self.img)
        else:
            self.head = backbone.inference(self.is_training, self.img)

    def compute_loss(self):
        with tf.name_scope('loss_0'):
            cls_loss = loss(self.head, self.truth)
            self.all_loss = cls_loss + self.l2_loss
        return self.all_loss

    def predict(self):
        max_score = tf.argmax(tf.reshape(self.head, [-1, 4, cfg.classes]), 2)
        return max_score
