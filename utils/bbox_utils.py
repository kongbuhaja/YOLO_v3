import tensorflow as tf
import numpy as np

def bbox_iou(bbox1, bbox2, xywh=True):
    if xywh:
        area1 = bbox1[..., 2] * bbox1[..., 3]
        area2 = bbox2[..., 2] * bbox2[..., 3]
        bbox1 = tf.concat([bbox1[..., :2] - bbox1[..., 2:] * 0.5, bbox1[..., :2] + bbox1[..., 2:] * 0.5], -1)
        bbox2 = tf.concat([bbox2[..., :2] - bbox2[..., 2:] * 0.5, bbox2[..., :2] + bbox2[..., 2:] * 0.5], -1)

    else:
        area1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
        area2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])
    
    Left_Top = tf.maximum(bbox1[..., :2], bbox2[..., :2])
    Right_Bottom = tf.minimum(bbox1[..., 2:], bbox2[..., 2:])

    inter_section = tf.maximum(Right_Bottom - Left_Top, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = tf.maximum(area1 + area2 - inter_area, 1e-6)

    return inter_area / union_area

def xyxy_to_xywh(boxes, with_label=False):
    labels = tf.concat([(boxes[..., 0:2] + boxes[..., 2:4])*0.5, boxes[..., 2:4] - boxes[..., 0:2]],-1)
    if with_label:
        labels = tf.concat([labels, boxes[..., 4:5]], -1)
    return labels

def xywh_to_xyxy(boxes, with_label=False):
    labels = tf.concat([boxes[..., :2] - boxes[..., 2:4] * 0.5 , boxes[..., :2] + boxes[..., 2:4] * 0.5], -1)
    if with_label:
        labels = tf.concat([labels, boxes[..., 4:5]], -1)
    return labels

def normalize_bbox(w, h, bbox):
    bbox[..., [0,2]] /= w
    bbox[..., [1,3]] /= h
    return bbox

def bbox_iou_wh(wh1, wh2):
    inter_section = tf.minimum(wh1, wh2)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = tf.maximum(wh1[..., 0] * wh1[..., 1] + wh2[..., 0] * wh2[..., 1] - inter_area, 1e-6)
    return inter_area / union_area

def bbox_iou_wh_np(wh1, wh2):
    inter_section = np.minimum(wh1, wh2)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = np.maximum(wh1[..., 0] * wh1[..., 1] + wh2[..., 0] * wh2[..., 1] - inter_area, 1e-6)
    return inter_area / union_area