import tensorflow as tf
from losses.common import *
from utils import bbox_utils

@tf.function
def v3_conf_loss(label_xywhc, pred_xywhc_raw, anchor, iou_threshold, inf, eps, noobj):
    pred_xy_iou = tf.sigmoid(pred_xywhc_raw[..., :2]) + anchor[..., :2]
    pred_wh_iou = tf.exp(pred_xywhc_raw[..., 2:4]) * anchor[..., 2:]
    pred_xywh_iou = tf.concat([pred_xy_iou, pred_wh_iou], -1)
    
    label_xywh_iou = label_xywhc[..., :4]
    
    ious = bbox_utils.bbox_iou(pred_xywh_iou, label_xywh_iou)[..., None]
    
    pred_conf = tf.sigmoid(pred_xywhc_raw[..., 4:5])
    resp_mask = label_xywhc[..., 4:5]
    noresp_mask = (1.0 - resp_mask) * tf.cast(ious < iou_threshold, dtype=tf.float32)
    
    obj_loss = tf.minimum(tf.reduce_sum(resp_mask * binaryCrossEntropy(resp_mask, pred_conf, eps), [1,2,3,4]), inf)
    noobj_loss = tf.minimum(noobj * tf.reduce_sum(noresp_mask * binaryCrossEntropy(resp_mask, pred_conf, eps), [1,2,3,4]), inf)
    
    return obj_loss + noobj_loss

@tf.function
def v2_conf_loss(label_xywhc, pred_xywhc_raw, anchor, iou_threshold, inf, eps, noobj):
    pred_xy_iou = tf.sigmoid(pred_xywhc_raw[..., :2]) + anchor[..., :2]
    pred_wh_iou = tf.exp(pred_xywhc_raw[..., 2:4]) * anchor[..., 2:]
    pred_xywh_iou = tf.concat([pred_xy_iou, pred_wh_iou], -1)
    
    label_xywh_iou = label_xywhc[..., :4]
    
    ious = bbox_utils.bbox_iou(pred_xywh_iou, label_xywh_iou)[..., None]
    
    pred_conf = tf.sigmoid(pred_xywhc_raw[..., 4:5])
    resp_mask = label_xywhc[..., 4:5]
    noresp_mask = (1.0 - resp_mask) * tf.cast(ious < iou_threshold, dtype=tf.float32)
    
    obj_loss = tf.minmum(tf.reduce_sum(resp_mask * tf.square((resp_mask - pred_conf)), [1,2,3,4]), inf)
    noobj_loss = tf.minimum(tf.reduce_sum(noobj * noresp_mask * tf.square((resp_mask - pred_conf)), [1,2,3,4]), inf)
    
    return obj_loss + noobj_loss