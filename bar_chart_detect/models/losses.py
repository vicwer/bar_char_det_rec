#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys
sys.path.append('..')
from bar_chart_detect.config import cfg

class bar_det:
    '''
    convert final layer features to bounding box parameters.

        parameters
        ----------
        feats : tensor
            final convolutional layer features.
        ahchors : array-like
            anchor box widths and heights
        num_classes : int
            number of target classes

        returns
        -------
        box_xy : tensor
            x, y box predictions adjusted by spatial location in conv layer.
        box_wh : tensor
            w, h box predictions adjusted by anchors and conv spatial resolution
        box_conf : tensor
            probability estimate for whether each box contains any object
        box_class_pred : tnesor
            probability distribution estimate for each box over class labels
    '''
    def __init__(self, anchors, num_classes, img_shape):
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_shape = img_shape

    def build(self, feats):
        # reshape to batch, height, width, num_anchors, box_params
        anchors_tensor = tf.reshape(self.anchors, [1,1,1,cfg.num_anchors_per_layer,2])

        # dynamic implementation of conv dims for fully convolutional model
        conv_dims = tf.stack([tf.shape(feats)[2], tf.shape(feats)[1]]) # assuming channels last, w h
        # the height index is the inner most iteration
        conv_height_index = tf.range(conv_dims[1])
        conv_width_index = tf.range(conv_dims[0])
        conv_width_index, conv_height_index = tf.meshgrid(conv_width_index, conv_height_index)
        conv_height_index = tf.reshape(conv_height_index, [-1, 1])
        conv_width_index = tf.reshape(conv_width_index, [-1, 1])
        conv_index = tf.concat([conv_width_index, conv_height_index], axis=-1)
        # 0, 0
        # 1, 0
        # 2, 0
        # ...
        # channel-1, 0
        # 0, 1
        # 1, 1
        # 2, 1
        # ...
        # channel-1, 1
        conv_index = tf.reshape(conv_index, [1, conv_dims[1], conv_dims[0], 1, 2])  # [1, 13, 13, 1, 2]
        conv_index = tf.cast(conv_index, tf.float32)

        feats = tf.reshape(feats, [-1, conv_dims[1], conv_dims[0], cfg.num_anchors_per_layer, self.num_classes + 5])

        conv_dims = tf.cast(tf.reshape(conv_dims, [1, 1, 1, 1, 2]), tf.float32)
        img_dims = tf.stack([self.img_shape[2], self.img_shape[1]])   # w, h
        img_dims = tf.cast(tf.reshape(img_dims, [1, 1, 1, 1, 2]), tf.float32)

        box_xy = tf.sigmoid(feats[..., :2])  # σ(tx), σ(ty)     # [None, 13, 13, 3, 2]
        box_twh = feats[..., 2:4]
        box_wh = tf.exp(box_twh)  # exp(tw), exp(th)    # [None, 13, 13, 3, 2]
        self.box_confidence = tf.sigmoid(feats[..., 4:5])
        self.box_class_probs = tf.sigmoid(feats[..., 5:])        # multi-label classification
        self.box_xy = (box_xy + conv_index) / conv_dims  # relative the whole img [0, 1]
        self.box_wh = box_wh * anchors_tensor / img_dims  # relative the whole img [0, 1]
        self.loc_txywh = tf.concat([box_xy, box_twh], axis=-1)

        # box_xy: [None, 13, 13, 3, 2]
        # box_wh: [None, 13, 13, 3, 2]
        # box_confidence: [None, 13, 13, 3, 1]
        # box_class_probs: [None, 13, 13, 3, 20]
        return self.box_xy, self.box_wh, self.box_confidence, self.box_class_probs, self.loc_txywh

def preprocess_true_boxes(true_boxes, anchors, feat_size, image_size):
    """

    :param true_boxes: x, y, w, h, class
    :param anchors:
    :param feat_size:
    :param image_size:
    :return:
    """
    num_anchors = cfg.num_anchors_per_layer

    true_wh = tf.expand_dims(true_boxes[..., 2:4], 2)  # [batch, 30, 1, 2]
    true_wh_half = true_wh / 2.
    true_mins = 0 - true_wh_half
    true_maxes = true_wh_half

    img_wh = tf.reshape(tf.stack([image_size[2], image_size[1]]), [1, -1])
    anchors = anchors / tf.cast(img_wh, tf.float32)  # normalize
    anchors_shape = tf.shape(anchors)  # [num_anchors, 2]
    anchors = tf.reshape(anchors, [1, 1, anchors_shape[0], anchors_shape[1]])  # [1, 1, num_anchors, 2]
    anchors_half = anchors / 2.
    anchors_mins = 0 - anchors_half
    anchors_maxes = anchors_half

    intersect_mins = tf.maximum(true_mins, anchors_mins)
    intersect_maxes = tf.minimum(true_maxes, anchors_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)     # [batch, 30, num_anchors, 2]
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]   # [batch, 30, num_anchors]

    true_areas = true_wh[..., 0] * true_wh[..., 1]      # [batch, 30, 1]
    anchors_areas = anchors[..., 0] * anchors[..., 1]   # [1, 1, num_anchors]

    union_areas = true_areas + anchors_areas - intersect_areas  # [batch, 30, num_anchors]

    iou_scores = intersect_areas / union_areas  # [batch, 30, num_anchors]
    valid = tf.logical_not(tf.reduce_all(tf.equal(iou_scores, 0), axis=-1))     # [batch, 30]
    iout_argmax = tf.cast(tf.argmax(iou_scores, axis=-1), tf.int32)   # [batch, 30], (0, 1, 2)
    anchors = tf.reshape(anchors, [-1, 2])      # has been normalize by img shape
    anchors_cf = tf.gather(anchors, iout_argmax)   # [batch, 30, 2]

    feat_wh = tf.reshape(tf.stack([feat_size[2], feat_size[1]]), [1, -1])  # (1, 2)
    cxy = tf.cast(tf.floor(true_boxes[..., :2] * tf.cast(feat_wh, tf.float32)),
                  tf.int32)    # [batch, 30, 2]   bx = cx + σ(tx)
    sig_xy = tf.cast(true_boxes[..., :2] * tf.cast(feat_wh, tf.float32) - tf.cast(cxy, tf.float32),
                     tf.float32)   # [batch, 30, 2]
    idx = cxy[..., 1] * (num_anchors * feat_size[2]) + num_anchors * cxy[..., 0] + iout_argmax  # [batch, 30]
    idx_one_hot = tf.one_hot(idx, depth=feat_size[1] * feat_size[2] * num_anchors)   # [batch, 30, 13x13x3]
    idx_one_hot = tf.reshape(idx_one_hot,
                        [-1, cfg.train.max_truth, feat_size[1], feat_size[2], num_anchors,
                         1])  # (batch, 30, 13, 13, 3, 1)
    loc_scale = 2 - true_boxes[..., 2] * true_boxes[..., 3]     # (batch, 30)
    mask = []
    loc_cls = []
    scale = []
    for i in range(cfg.batch_size):
        idx_i = tf.where(valid[i])[:, 0]    # (?, )    # false / true
        mask_i = tf.gather(idx_one_hot[i], idx_i)   # (?, 13, 13, 3, 1)

        scale_i = tf.gather(loc_scale[i], idx_i)    # (?, )
        scale_i = tf.reshape(scale_i, [-1, 1, 1, 1, 1])     # (?, 1, 1, 1, 1)
        scale_i = scale_i * mask_i      # (?, 13, 13, 3, 1)
        scale_i = tf.reduce_sum(scale_i, axis=0)        # (13, 13, 3, 1)
        scale_i = tf.maximum(tf.minimum(scale_i, 2), 1)
        scale.append(scale_i)

        true_boxes_i = tf.gather(true_boxes[i], idx_i)    # (?, 5)
        sig_xy_i = tf.gather(sig_xy[i], idx_i)    # (?, 2)
        anchors_cf_i = tf.gather(anchors_cf[i], idx_i)    # (?, 2)
        twh = tf.log(true_boxes_i[:, 2:4] / anchors_cf_i)
        loc_cls_i = tf.concat([sig_xy_i, twh, true_boxes_i[:, -1:]], axis=-1)    # (?, 5)
        loc_cls_i = tf.reshape(loc_cls_i, [-1, 1, 1, 1, 5])     # (?, 1, 1, 1, 5)
        loc_cls_i = loc_cls_i * mask_i      # (?, 13, 13, 3, 5)
        loc_cls_i = tf.reduce_sum(loc_cls_i, axis=[0])  # (13, 13, 3, 5)
        # exception, one anchor is responsible for 2 or more object
        loc_cls_i = tf.concat([loc_cls_i[..., :4], tf.minimum(loc_cls_i[..., -1:], 19)], axis=-1)
        loc_cls.append(loc_cls_i)

        mask_i = tf.reduce_sum(mask_i, axis=[0])    # (13, 13, 3, 1)
        mask_i = tf.minimum(mask_i, 1)
        mask.append(mask_i)

    loc_cls = tf.stack(loc_cls, axis=0)     # (σ(tx), σ(tx), tw, th, cls)
    mask = tf.stack(mask, axis=0)
    scale = tf.stack(scale, axis=0)
    return loc_cls, mask, scale


def confidence_loss(pred_xy, pred_wh, pred_confidence, true_boxes, detectors_mask):
    """

    :param pred_xy: [batch, 13, 13, 5, 2] from plate_det
    :param pred_wh: [batch, 13, 13, 5, 2] from plate_det
    :param pred_confidence: [batch, 13, 13, 5, 1] from plate_det
    :param true_boxes: [batch, 30, 5]
    :param detectors_mask: [batch, 13, 13, 5, 1]
    :return:
    """
    pred_xy = tf.expand_dims(pred_xy, 4)  # [batch, 13, 13, 3, 1, 2]
    pred_wh = tf.expand_dims(pred_wh, 4)  # [batch, 13, 13, 3, 1, 2]

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_boxes_shape = tf.shape(true_boxes)  # [batch, num_true_boxes, box_params(5)]
    true_boxes = tf.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
        ])  # [batch, 1, 1, 1, num_true_boxes, 5]
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    # Find IOU of each predicted box with each ground truth box.
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    # [batch, 13, 13, 3, 1, 2] [batch, 1, 1, 1, num_true_boxes, 2]
    intersect_mins = tf.maximum(pred_mins, true_mins)
    # [batch, 13, 13, 3, num_true_boxes, 2]
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    # [batch, 13, 13, 3, num_true_boxes, 2]
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    # [batch, 13, 13, 3, num_true_boxes, 2]
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    # [batch, 13, 13, 3, num_true_boxes]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    # [batch, 13, 13, 3, 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    # [batch, 1, 1, 1, num_true_boxes]
    union_areas = pred_areas + true_areas - intersect_areas
    # [batch, 13, 13, 3, num_true_boxes]
    iou_scores = intersect_areas / union_areas

    # Best IOUs for each loction.
    best_ious = tf.reduce_max(iou_scores, axis=-1, keepdims=True)  # Best IOU scores.
    # [batch, 13, 13, 3, 1]

    # A detector has found an object if IOU > thresh for some true box.
    object_ignore = tf.cast(best_ious > cfg.train.ignore_thresh, best_ious.dtype)
    no_object_weights = (1 - object_ignore) * (1 - detectors_mask)  # [batch, 13, 13, 5, 1]
    no_objects_loss = no_object_weights * tf.square(pred_confidence)
    objects_loss = detectors_mask * tf.square(1 - pred_confidence)

    objectness_loss = tf.reduce_sum(objects_loss + no_objects_loss)
    #objectness_loss = tf.reduce_mean(objects_loss + no_objects_loss)
    return objectness_loss


def cord_cls_loss(
                detectors_mask,
                matching_true_boxes,
                num_classes,
                pred_class_prob,
                pred_boxes,
                loc_scale,
              ):
    """
    :param detectors_mask: [batch, 13, 13, 3, 1]
    :param matching_true_boxes: [batch, 13, 13, 3, 5]   [σ(tx), σ(ty), tw, th, cls]
    :param num_classes: 20
    :param pred_class_prob: [batch, 13, 13, 3, 20]
    :param pred_boxes: [batch, 13, 13, 3, 4]
    :param loc_scale: [batch, 13, 13, 3, 1]
    :return:
        mean_loss: float
        mean localization loss across minibatch
    """

    # Classification loss for matching detections.
    # NOTE: does not use categorical cross-entropy loss here.
    matching_classes = tf.cast(matching_true_boxes[..., 4], tf.int32)   # [batch, 13, 13, 3]
    matching_classes = tf.one_hot(matching_classes, num_classes)    # [batch, 13, 13, 3, 20]
    classification_loss = (detectors_mask *
                           tf.square(matching_classes - pred_class_prob))   # [batch, 13, 13, 3, 20]

    # Coordinate loss for matching detection boxes.   [σ(tx), σ(ty), tw, th]
    matching_boxes = matching_true_boxes[..., 0:4]
    coordinates_loss = (detectors_mask * loc_scale * tf.square(matching_boxes - pred_boxes))

    classification_loss_sum = tf.reduce_sum(classification_loss)
    #classification_loss_sum = tf.reduce_mean(classification_loss)
    coordinates_loss_sum = tf.reduce_sum(coordinates_loss)
    #coordinates_loss_sum = tf.reduce_mean(coordinates_loss)

    return classification_loss_sum + coordinates_loss_sum

