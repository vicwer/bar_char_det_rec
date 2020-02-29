from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
import os

cfg = edict()

cfg.classes = 19

cfg.names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '$']
cfg.batch_size = 512

cfg.data_path = '../data/train_data/train.records'
cnt_path = os.getcwd()
cfg.ckpt_path = cnt_path + '/number_recognize'
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
cfg.train.max_truth = 1
cfg.train.image_resized = 608
cfg.train.num_gpus = 4
cfg.train.tower = 'tower'
