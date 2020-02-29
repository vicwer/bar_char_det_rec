#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys
sys.path.append('..')
from number_recognize.config import cfg

def loss(preds, labels):
    pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(preds, [-1, cfg.classes]), labels=tf.reshape(labels, [-1, cfg.classes]))
    pred_loss = tf.reshape(pred_loss, [-1, 4])
    pred_loss = tf.reduce_sum(pred_loss, 1)
    pred_loss = tf.reduce_sum(pred_loss)
    return pred_loss
