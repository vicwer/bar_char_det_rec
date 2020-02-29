#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys
sys.path.append('..')
from bar_chart_detect.models.network import Network
from bar_chart_detect.config import cfg
from bar_chart_detect.models.losses import bar_det, preprocess_true_boxes, confidence_loss, cord_cls_loss

class BarDetNet:
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
        with tf.variable_scope('detection'):
            self.anchors = tf.constant(cfg.anchors, dtype=tf.float32)
            det = bar_det(self.anchors, cfg.classes, self.img_shape)
            self.pred_xy, self.pred_wh, self.pred_confidence, self.pred_class_prob, \
                    self.loc_txywh = det.build(self.head)

    def compute_loss(self):
        with tf.name_scope('loss_0'):
            matching_true_boxes, detectors_mask, loc_scale = \
                    preprocess_true_boxes(self.truth, self.anchors, tf.shape(self.head), self.img_shape)
            objectness_loss = confidence_loss(self.pred_xy, self.pred_wh, self.pred_confidence, self.truth, detectors_mask)
            cord_loss = cord_cls_loss(detectors_mask, matching_true_boxes, cfg.classes, \
                                self.pred_class_prob, self.loc_txywh, loc_scale)
            self.loss = objectness_loss + cord_loss + self.l2_loss
        return self.loss

    def predict(self, img_hw, iou_threshold=0.5, score_threshold=0.5):
        '''
        only support single image prediction
        '''
        img_hwhw = tf.expand_dims(tf.stack([img_hw[0], img_hw[1]]*2, axis=0), axis=0)
        with tf.name_scope('predict'):
            # (y1, x1, y2, x2)
            pred_loc = tf.concat([self.pred_xy[..., 1:] - 0.5 * self.pred_wh[..., 1:],
                                  self.pred_xy[..., 0:1] - 0.5 * self.pred_wh[..., 0:1],
                                  self.pred_xy[..., 1:] + 0.5 * self.pred_wh[..., 1:],
                                  self.pred_xy[..., 0:1] + 0.5 * self.pred_wh[..., 0:1]], axis=-1)
            pred_loc = tf.maximum(tf.minimum(pred_loc, 1), 0)
            self.pred_loc = tf.reshape(pred_loc, [-1, 4]) * img_hwhw
            self.pred_obj = tf.reshape(self.pred_confidence, shape=[-1])
            self.pred_cls = tf.reshape(self.pred_class_prob, [-1, cfg.classes])

        # score filter
        box_scores = tf.expand_dims(self.pred_obj, axis=1) * self.pred_cls
        box_label = tf.argmax(box_scores, axis=-1)
        box_scores_max = tf.reduce_max(box_scores, axis=-1)

        pred_mask = box_scores_max > score_threshold
        boxes = tf.boolean_mask(self.pred_loc, pred_mask)
        scores = tf.boolean_mask(box_scores_max, pred_mask)
        classes = tf.boolean_mask(box_label, pred_mask)

        idx_nms = tf.image.non_max_suppression(boxes, scores, max_output_size=30, iou_threshold=iou_threshold)
        boxes = tf.gather(boxes, idx_nms)
        scores = tf.gather(scores, idx_nms)
        classes = tf.gather(classes, idx_nms)

        return boxes, scores, classes
