from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
import os

cfg = edict()

cfg.classes = 2
cfg.num_anchors = 4
cfg.num_anchors_per_layer = 4
cfg.num = 4
cfg.anchors = np.array([[17, 24], [48, 24], [50, 150], [60, 400]])

cfg.names = ['bar', 'coord']
cfg.batch_size = 32

cfg.data_path = '../data/train_data/train.records'
cnt_path = os.getcwd()
cfg.ckpt_path = cnt_path + '/bar_chart_detect'
cfg.ckpt_path_608 = cfg.ckpt_path + '/ckpt/'

# training options
cfg.train = edict()

cfg.train.ignore_thresh = .5
cfg.train.momentum = 0.9
cfg.train.bn_training = True
cfg.train.weight_decay = 0.0005
cfg.train.learning_rate = [1e-3, 1e-4, 1e-5]
cfg.train.max_batches = 60010
cfg.train.lr_steps = [40000., 50000.]
cfg.train.lr_scales = [.1, .1]
cfg.train.max_truth = 30
cfg.train.mask = np.array([[0, 1, 2, 3]])
cfg.train.image_resized = 608
cfg.train.num_gpus = 2
cfg.train.tower = 'tower'
