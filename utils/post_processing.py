import tensorflow as tf
import numpy as np
from config import *
from utils import bbox_utils

def prediction_to_bbox(grids, anchors):
    batch_size = grids[0].shape[0]
    bboxes = tf.zeros((batch_size,0,4))
    scores = tf.zeros((batch_size, 0))
    probs = tf.zeros((batch_size, 0))
    for grid, anchor, stride in zip(grids, anchors, STRIDES):
        grid = tf.reshape(grid, [batch_size, -1, 5+NUM_CLASSES])

        xy = tf.sigmoid(grid[..., :2]) + anchor[..., :2]
        wh = tf.exp(grid[..., 2:4]) * anchor[..., 2:4]
        score = tf.sigmoid(grid[..., 4])
        prob = tf.sigmoid(grid[..., 5:])

        bboxes = tf.concat([bboxes, tf.concat([xy, wh],-1)*stride], 1)
        bboxes = bbox_utils.xywh_to_xyxy(bboxes)
        scores = tf.concat([scores, score], 1)
        probs = tf.concat([probs, prob], 1)

    # score_argsort = tf.argsort(scores, direction='DESCENDING', axis=-1)

    # bboxes = tf.gather(bboxes, score_argsort, batch_dims=1)
    # scores = tf.gather(scores, score_argsort, batch_dims=1)
    # classes = tf.gather(classes, score_argsort, batch_dims=1)

    return bboxes, scores, classes
    
def NMS_(bboxes, scores, classes, score_threshold=0.6):
    NMS_bboxes = tf.zeros((0,4))
    NMS_scores = tf.zeros((0))
    NMS_classes = tf.zeros((0), tf.int32)

    score_mask = tf.cast(scores > score_threshold, tf.int32)
    positive_count = tf.reduce_sum(score_mask)
    score_argsort = tf.argsort(scores, direction='DESCENDING')

    bboxes = tf.gather(bboxes, score_argsort)[:positive_count]
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


def NMS(bboxes, scores, probs, score_threshold=SCORE_THRESHOLD, iou_threshold=IOU_THRESHOLD, sigma=SIGMA, method=NMS_TYPE):
    NMS_bboxes = tf.zeros((0, 4))
    NMS_scores = tf.zeros((0))
    NMS_probs = tf.zeros((0, 3))
    classes = tf.argmax(probs, -1)
    unique_classes = tf.unique(classes)[0]

    for unique_class in unique_classes:
        class_idx = tf.squeeze(tf.where(tf.logical_and(classes == unique_class, scores >= score_threshold)))
        target_bboxes = tf.gather(bboxes, class_idx)
        target_scores = tf.gather(scores, class_idx)
        target_probs = tf.gather(probs, class_idx)
        while(target_scores.shape[0]):
            indices = tf.argsort(target_scores, direction='DESCENDING')
            max_bbox = target_bboxes[indices[0]][None]
            max_score = target_scores[indices[0]][None]
            max_prob = target_probs[indices[0]][None]

            NMS_bboxes = tf.concat([NMS_bboxes, max_bbox], 0)
            NMS_scores = tf.concat([NMS_scores, max_score], 0)
            NMS_probs = tf.concat([NMS_probs, max_prob], 0)
            target_bboxes = tf.gather(target_bboxes, indices[1:])
            target_scores = tf.gather(target_scores, indices[1:])
            target_probs = tf.gather(target_probs, indices[1:])
            
            ious = bbox_utils.bbox_iou(max_bbox, target_bboxes, False)
            if method == 'normal':
                target_scores = tf.where(ious > iou_threshold, 0, target_scores)
            elif method == 'soft_normal':
                target_scores = tf.where(ious > iou_threshold, target_scores * (1 - ious), target_scores)
            elif method == 'soft_gaussian':
                target_scores = tf.exp(-(ious)**2/sigma) * target_scores

    filter = tf.squeeze(tf.where(NMS_scores > score_threshold))
    NMS_bboxes = tf.gather(NMS_bboxes, filter)
    NMS_scores = tf.gather(NMS_scores, filter)
    NMS_probs = tf.gather(NMS_probs, filter)
    NMS_classes = tf.argmax(NMS_probs, -1)
    
    return NMS_bboxes, NMS_scores, NMS_classes