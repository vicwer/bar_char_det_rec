#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import sys
sys.path.append('..')
import numpy as np
from number_recognize.config import cfg

def network_arg_scope(
        is_training=True, weight_decay=cfg.train.weight_decay, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=False):
    batch_norm_params = {
        'is_training': is_training, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
        #'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
        'trainable': cfg.train.bn_training,
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu6,
            #activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params,
            padding='SAME'):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

class Network(object):
    def __init__(self):
        pass

    def inference(self, mode, inputs, scope='NumRecNet'):
        is_training = mode
        print(inputs)
        with slim.arg_scope(network_arg_scope(is_training=is_training)):
            with tf.variable_scope(scope, reuse=False):
                conv0 = conv2d(inputs, 32, 1, name='conv_0')
                conv2 = conv2d(conv0, 32, 1, name='conv_2')
                conv3 = conv2d(conv2, 32, 1, name='conv_3')
                route4 = route([conv2, conv3], name='route_4')
                conv5 = conv2d(route4, 64, 2, name='conv_5')
                conv6 = conv2d(conv5, 64, 1, name='conv_6')
                conv7 = conv2d(conv6, 64, 1, name='conv_7')
                route8 = route([conv7, conv5], name='route_8')
                conv9 = conv2d(route8, 128, 1, name='conv_9')
                conv10 = conv2d(conv9, 128, 1, name='conv_10')
                route11 = route([conv10, conv7], name='route_11')
                conv12 = conv2d(route11, 256, 2, name='conv_12')
                conv13 = conv2d(conv12, 256, 1, name='conv_13')
                conv14 = conv2d(conv13, 256, 1, name='conv_14')
                conv15 = conv2d(conv14, 256, 2, name='conv_15')
                conv16 = conv2d(conv15, 256, 1, name='conv_16')
                pool17 = avg_pool(conv16, [3, 8], name='pool_17')
                logits = tf.squeeze(pool17, [1, 2], name='logits')
                print('logits', logits.get_shape())
                fc1 = fully_connected(logits, cfg.classes, name='fc1')
                fc2 = fully_connected(logits, cfg.classes, name='fc2')
                fc3 = fully_connected(logits, cfg.classes, name='fc3')
                fc4 = fully_connected(logits, cfg.classes, name='fc4')
                fc = tf.concat([fc1, fc2, fc3, fc4], 1, name='fc_concat')
                print('fc', fc.get_shape())
                if is_training:
                    #l2_loss = tf.add_n(slim.losses.get_regularization_losses())
                    l2_loss = tf.add_n(tf.losses.get_regularization_losses())
                    return fc, l2_loss
                else:
                    return fc

def conv2d(inputs, c_outputs, s, name):
    output = slim.conv2d(inputs, num_outputs=c_outputs, kernel_size=[3,3], stride=s, scope=name)
    print(name, output.get_shape())
    return output

def route(input_list, name):
    with tf.name_scope(name):
        output = tf.concat(input_list, 3, name='concat')
    print(name, output.get_shape())
    return output

def maxpool2x2(input, name):
    output = slim.max_pool2d(input, kernel_size=[2, 2], stride=2, scope=name)
    print(name, output.get_shape())
    return output

def unpool2x2(input, name):
    with tf.name_scope(name):
        out = tf.concat([input, tf.zeros_like(input)], 3, name='concat_1')
        output = tf.concat([out, tf.zeros_like(out)], 2, name='concat_2')
        n, h, w, c = input.get_shape()[0], input.get_shape()[1], input.get_shape()[2], input.get_shape()[3]
        res = tf.reshape(output, (-1, h*2, w*2, c))
        print(name, res.get_shape())
    return res

def avg_pool(input, kernel_size, name):
    output = slim.avg_pool2d(input, kernel_size, scope=name)
    print(name, output.get_shape())
    return output

def fully_connected(input, c_outputs, name):
    output = slim.fully_connected(input, c_outputs, activation_fn=None, scope=name)
    print(name, output.get_shape())
    return output
