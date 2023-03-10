import tensorflow as tf
import numpy as np
from config import *
from utils import bbox_utils

def prediction_to_bbox(grids, anchors):
    batch_size = grids[0].shape[0]
    bboxes = tf.zeros((batch_size,0,4))
    scores = tf.zeros((batch_size, 0))
    classes = tf.zeros((batch_size, 0), tf.int32)
    for grid, anchor, stride in zip(grids, anchors, STRIDES):
        grid = tf.reshape(grid, [batch_size, -1, 25])

        xy = tf.sigmoid(grid[..., :2]) + anchor[..., :2]
        wh = tf.exp(grid[..., 2:4]) * anchor[..., 2:4]
        prob = tf.sigmoid(grid[..., 5:])
        cls = tf.cast(tf.argmax(prob, -1), tf.int32)
        score = tf.reduce_max(prob, -1) * tf.sigmoid(grid[..., 4])

        bboxes = tf.concat([bboxes, tf.concat([xy, wh],-1)*stride], 1)
        scores = tf.concat([scores, score], 1)
        classes = tf.concat([classes, cls], 1)

    score_argsort = tf.argsort(scores, direction='DESCENDING', axis=-1)

    bboxes = tf.gather(bboxes, score_argsort, batch_dims=1)
    scores = tf.gather(scores, score_argsort, batch_dims=1)
    classes = tf.gather(classes, score_argsort, batch_dims=1)

    return bboxes, scores, classes
    
def NMS(bboxes, scores, classes, score_threshold=0.6):
    NMS_bboxes = tf.zeros((0,4))
    NMS_scores = tf.zeros((0))
    NMS_classes = tf.zeros((0), tf.int32)

    score_mask = tf.cast(scores > score_threshold, tf.int32)
    positive_count = tf.reduce_sum(score_mask)
    score_argsort = tf.argsort(scores, direction='DESCENDING')

    bboxes = bbox_utils.xywh_to_xyxy(tf.gather(bboxes, score_argsort)[:positive_count])
    bboxes = tf.minimum(tf.maximum(0, bboxes), IMAGE_SIZE)
    scores = tf.gather(scores, score_argsort)[:positive_count]
    classes = tf.gather(classes, score_argsort)[:positive_count]
    unique_classes, idxs = tf.unique(classes)

    for u_class in unique_classes:
        class_mask = tf.cast(classes == u_class, tf.int32)
        class_count = tf.reduce_sum(class_mask)
        sample_argsort = tf.argsort(class_mask, direction='DESCENDING')[:class_count]

        while(sample_argsort.shape[0]>0):
            index = sample_argsort[0]
            NMS_bboxes = tf.concat([NMS_bboxes, bboxes[index][None]], 0)
            NMS_scores = tf.concat([NMS_scores, scores[index][None]], 0)
            NMS_classes = tf.concat([NMS_classes, classes[index][None]], 0)

            ious = bbox_utils.bbox_iou(bboxes[index], tf.gather(bboxes, sample_argsort), xywh=False)
            iou_filter_mask = tf.cast(ious > 0.4, tf.int32)
            remove_count = tf.reduce_sum(iou_filter_mask)
            iou_argsort = tf.argsort(ious, direction='DESCENDING')
            sample_argsort = tf.gather(sample_argsort, iou_argsort)[remove_count:]

    return NMS_bboxes, NMS_scores, NMS_classes


# ????????? ???????????????
def soft_NMS(bboxes, scores, classes, score_threshold=0.6):
    NMS_bboxes = tf.zeros((0,4))
    NMS_scores = tf.zeros((0))
    NMS_classes = tf.zeros((0), tf.int32)

    score_mask = tf.cast(scores > score_threshold, tf.int32)
    positive_count = tf.reduce_sum(score_mask)
    score_argsort = tf.argsort(scores, direction='DESCENDING')

    bboxes = bbox_utils.xywh_to_xyxy(tf.gather(bboxes, score_argsort)[:positive_count])
    bboxes = tf.minimum(tf.maximum(0, bboxes), IMAGE_SIZE)
    scores = tf.gather(scores, score_argsort)[:positive_count]
    classes = tf.gather(classes, score_argsort)[:positive_count]
    unique_classes, idxs = tf.unique(classes)

    for u_class in unique_classes:
        class_mask = tf.cast(classes == u_class, tf.int32)
        class_count = tf.reduce_sum(class_mask)
        sample_argsort = tf.argsort(class_mask, direction='DESCENDING')[:class_count]

        while(sample_argsort.shape[0]>0):
            index = sample_argsort[0]
            NMS_bboxes = tf.concat([NMS_bboxes, bboxes[index][None]], 0)
            NMS_scores = tf.concat([NMS_scores, scores[index][None]], 0)
            NMS_classes = tf.concat([NMS_classes, classes[index][None]], 0)

            ious = bbox_utils.bbox_iou(bboxes[index], tf.gather(bboxes, sample_argsort), xywh=False)
            iou_filter_mask = tf.cast(ious > 0.4, tf.int32)
            remove_count = tf.reduce_sum(iou_filter_mask)
            iou_argsort = tf.argsort(ious, direction='DESCENDING')
            sample_argsort = tf.gather(sample_argsort, iou_argsort)[remove_count:]

    return NMS_bboxes, NMS_scores, NMS_classes