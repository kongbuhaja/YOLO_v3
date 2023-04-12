import tensorflow as tf
from utils import bbox_utils
from losses.common import *
from losses.loc_loss import *
from losses.conf_loss import *
from losses.prob_loss import *


@tf.function
def v3_loss(labels, preds, anchors, iou_threshold, inf=1e+30, eps=1e-7, coord=5, noobj=0.5):
    loc_loss, conf_loss, prob_loss = 0., 0., 0.

    for label, pred, anchor in zip(labels, preds, anchors):       
        loc_loss += v3_loc_loss(label[..., :5], pred[..., :4], anchor, inf, eps, coord)
        conf_loss += v3_conf_loss(label[..., :5], pred[..., :5], anchor, iou_threshold, inf, eps, noobj)
        prob_loss += v3_prob_loss(label[..., 4:], pred[..., 5:], inf, eps)
            
    loc_loss = tf.reduce_mean(loc_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    prob_loss = tf.reduce_mean(prob_loss)
    total_loss = loc_loss + conf_loss + prob_loss

    return loc_loss, conf_loss, prob_loss, total_loss

@tf.function
def v3_paper_loss(labels, preds, anchors, iou_threshold, inf=1e+30, eps=1e-7, coord=5, noobj=0.5):
    loc_loss, conf_loss, prob_loss = 0., 0., 0.

    for label, pred, anchor in zip(labels, preds, anchors):
        loc_loss += v3_paper_loc_loss(label[..., :5], pred[..., :4], anchor, inf, eps, coord)
        conf_loss += v3_conf_loss(label[..., :5], pred[..., :5], anchor, iou_threshold, inf, eps, noobj)
        prob_loss += v3_prob_loss(label[..., 4:], pred[..., 5:], inf, eps)
        
        
    loc_loss = tf.reduce_mean(loc_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    prob_loss = tf.reduce_mean(prob_loss)
    total_loss = loc_loss + conf_loss + prob_loss
    return loc_loss, conf_loss, prob_loss, total_loss

@tf.function
def v2_loss(labels, preds, anchors, strides, iou_threshold, inf=1e+30, eps=1e-7, coord=5, noobj=0.5):
    loc_loss, conf_loss, prob_loss = 0., 0., 0.

    for label, pred, anchor, stride in zip(labels, preds, anchors, strides):
        loc_loss += v2_loc_loss(label[..., :5], pred[..., :4], anchor, inf, eps, coord)
        conf_loss += v2_conf_loss(label[..., :5], pred[..., :5], anchor, iou_threshold, inf, eps, noobj)
        prob_loss += v2_prob_loss(label[..., 4:], pred[..., 5:], inf, eps)
                
    loc_loss = tf.reduce_mean(loc_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    prob_loss = tf.reduce_mean(prob_loss)
    total_loss = loc_loss + conf_loss + prob_loss
    return loc_loss, conf_loss, prob_loss, total_loss