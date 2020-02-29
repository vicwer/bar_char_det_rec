#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
from models.run_net import NumRecNet
from prepare_data.gen_data_batch import gen_data_batch
from config import cfg
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
          List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    is_training = True

    # data pipeline
    imgs, true_boxes = gen_data_batch(cfg.data_path, cfg.batch_size*cfg.train.num_gpus)
    imgs_split = tf.split(imgs, cfg.train.num_gpus)
    true_boxes_split = tf.split(true_boxes, cfg.train.num_gpus)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.), trainable=False)
    lr = tf.train.piecewise_constant(global_step, cfg.train.lr_steps, cfg.train.learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(cfg.train.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (cfg.train.tower, i)) as scope:
                    model = NumRecNet(imgs_split[i], true_boxes_split[i], is_training)
                    loss = model.compute_loss()
                    tf.get_variable_scope().reuse_variables()
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
                    if i == 0:
                        current_loss = loss
                        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="NumRecNet")
    grads = average_gradients(tower_grads)
    with tf.control_dependencies(update_op):
        train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars_det)

    # GPU config
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Create a saver
    saver = tf.train.Saver()
    ckpt_dir = re.sub(r'examples/', '', cfg.ckpt_path_608)

    # init
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, ckpt_dir+str(50000)+'_plate.ckpt-'+str(50000+1))
    #gs = 50000
    #sess.run(tf.assign(global_step, gs))

    # running
    for i in range(0, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, current_loss])
        if(i % 1 == 0):
            print(i,': ', loss_)
        if i % 1000 == 0 and i < 10000:
            saver.save(sess, ckpt_dir+str(i)+'_plate.ckpt', global_step=global_step, write_meta_graph=False)
        if i % 10000 == 0:
            saver.save(sess, ckpt_dir+str(i)+'_plate.ckpt', global_step=global_step, write_meta_graph=False)


if __name__ == '__main__':
    train()
