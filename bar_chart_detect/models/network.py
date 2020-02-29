#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import sys
sys.path.append('..')
import numpy as np
from bar_chart_detect.config import cfg

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

    def inference(self, mode, inputs, scope='BarDetNet'):
        is_training = mode
        print(inputs)
        with slim.arg_scope(network_arg_scope(is_training=is_training)):
            with tf.variable_scope(scope, reuse=False):
                conv0 = conv2d(inputs, 32, 2, name='conv_0')
                pool1 = maxpool2x2(conv0, name='pool_1')
                conv2 = conv2d(pool1, 32, 1, name='conv_2')
                conv3 = conv2d(conv2, 32, 1, name='conv_3')
                route4 = route([pool1, conv2, conv3], name='route_4')
                conv5 = conv2d(route4, 32, 2, name='conv_5')
                conv6 = conv2d(conv5, 32, 1, name='conv_6')
                conv7 = conv2d(conv6, 32, 1, name='conv_7')
                route8 = route([conv7, conv5], name='route_8')
                conv9 = conv2d(route8, 32, 1, name='conv_9')
                conv10 = conv2d(conv9, 32, 1, name='conv_10')
                route11 = route([conv10, conv7], name='route_11')
                conv12 = conv2d(route11, 32, 1, name='conv_12')
                conv13 = conv2d(conv12, 32, 1, name='conv_13')
                route14 = route([conv13, conv10, conv5], name='route_14')
                conv15 = conv2d(route14, 32, 1, name='conv_15')
                conv16 = conv2d(conv15, 32, 1, name='conv_16')
                route17 = route([conv5, conv13, conv16], name='route_17')
                conv18 = conv2d(route17, 64, 2, name='conv_18')
                conv19 = conv2d(conv18, 64, 1, name='conv_19')
                conv20 = conv2d(conv19, 32, 1, name='conv_20')
                route21 = route([conv20, conv18], name='route_21')
                conv22 = conv2d(route21, 64, 1, name='conv_22')
                conv23 = conv2d(conv22, 32, 1, name='conv_23')
                route24 = route([conv23, conv20], name='route_24')
                conv25 = conv2d(route24, 64, 1, name='conv_25')
                conv26 = conv2d(conv25, 32, 1, name='conv_26')
                route27 = route([conv26, conv23, conv18], name='route27')
                conv28 = conv2d(route27, 32, 1, name='conv_28')
                conv29 = conv2d(conv28, 32, 1, name='conv_29')
                route30 = route([conv18, conv26, conv29], name='route_30')
                conv31 = conv2d(route30, 128, 2, name='conv_31')
                conv32 = conv2d(conv31, 128, 1, name='conv_32')
                conv33 = conv2d(conv32, 64, 1, name='conv_33')
                route34 = route([conv33, conv31], name='route_34')
                conv35 = conv2d(route34, 128, 1, name='conv_35')
                conv36 = conv2d(conv35, 64, 1, name='conv_36')
                route37 = route([conv36, conv33], name='route_37')
                conv38 = conv2d(route37, 128, 1, name='conv_38')
                conv39 = conv2d(conv38, 64, 1, name='conv_39')
                route40 = route([conv39, conv36, conv31], name='route_40')
                conv41 = conv2d(route40, 128, 1, name='conv_41')
                conv42 = conv2d(conv41, 64, 1, name='conv_42')
                unpool43 = unpool2x2(conv42, name='unpool_43')
                unpool45 = unpool2x2(conv39, name='unpool_45')
                unpool47 = unpool2x2(conv31, name='unpool_47')
                route48 = route([conv18, conv26, conv29, unpool43, unpool45, unpool47], name='route_48')
                conv49 = conv2d(route48, 128, 1, name='conv_49')
                unpool50 = unpool2x2(conv49, name='unpool_50')
                route51 = route([conv5, conv13, conv16, unpool50], name='route_51')
                conv52 = conv2d(route51, 64, 1, name='route51')
                unpool53 = unpool2x2(conv52, name='unpool_53')
                route54 = route([conv2, conv3, unpool53], name='route_54')
                conv55 = conv2d(route54, (cfg.classes+5)*cfg.num_anchors, 1, name='conv_55')
                conv56 = slim.conv2d(conv55, num_outputs=(cfg.classes+5)*cfg.num_anchors, kernel_size=[3,3], stride=1, scope='conv_56', activation_fn=None, normalizer_fn=None)
                print('conv56', conv56.get_shape())
                if is_training:
                    #l2_loss = tf.add_n(slim.losses.get_regularization_losses())
                    l2_loss = tf.add_n(tf.losses.get_regularization_losses())
                    return conv56, l2_loss
                else:
                    return conv56

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

