import tensorflow as tf
from utils import bbox_utils

def binaryCrossEntropy(label, pred, eps=1e-7):
    pred = tf.minimum(tf.maximum(pred, eps), 1-eps)
    return -(label*tf.math.log(pred) + (1.-label)*tf.math.log(1.-pred))

def loss1(labels, preds, anchors, strides, iou_threshold, inf=1e+30, eps=1e-7, coord=5, noobj=0.5):
    loc_loss, conf_loss, prob_loss = 0., 0., 0.
    batch_size = labels[0].shape[0]

    for label, pred, anchor, stride in zip(labels, preds, anchors, strides):
        pred_xy = pred[..., :2]
        pred_wh = pred[..., 2:4]
        pred_conf = tf.sigmoid(pred[..., 4:5])
        pred_prob = tf.sigmoid(pred[..., 5:])

        label_xywh_ = label[..., :4]
        resp_mask = label[..., 4:5]
        label_prob = label[..., 5:]
        
        sig_xy = tf.maximum(label[..., :2] - anchor[..., :2] , eps)
        label_xy = tf.math.log(sig_xy) - tf.math.log(1-sig_xy)
        label_wh = tf.math.log(tf.maximum(label[..., 2:4] / anchor[..., 2:], eps))            

        pred_xy_ = tf.sigmoid(pred[..., :2]) + anchor[..., :2]
        pred_wh_ = tf.exp(tf.minimum(pred[..., 2:4], inf)) * anchor[..., 2:]
        pred_xywh_ = tf.concat([pred_xy_, pred_wh_], -1)
        
        ious = tf.expand_dims(bbox_utils.bbox_iou(pred_xywh_, label_xywh_), -1)

        noresp_mask =  (1.0 - resp_mask) * tf.cast(ious < iou_threshold, dtype=tf.float32)
        
        xy_loss = coord * tf.reduce_sum(resp_mask * tf.square(label_xy - pred_xy))
        wh_loss = coord * tf.reduce_sum(resp_mask * tf.square(label_wh - pred_wh))
        loc_loss += xy_loss + wh_loss
        
        obj_loss = tf.reduce_sum(resp_mask * binaryCrossEntropy(resp_mask, pred_conf, eps))
        noobj_loss = noobj * tf.reduce_sum(noresp_mask * binaryCrossEntropy(resp_mask, pred_conf, eps))
        conf_loss += obj_loss + noobj_loss

        prob_loss += 5*tf.reduce_sum(resp_mask * binaryCrossEntropy(label_prob, pred_prob, eps))
        
    loc_loss /= batch_size
    conf_loss /= batch_size
    prob_loss /= batch_size
    total_loss = loc_loss + conf_loss + prob_loss
    return loc_loss, conf_loss, prob_loss, total_loss

def loss2(labels, preds, anchors, strides, iou_threshold, inf=1e+30, eps=1e-7, coord=5, noobj=0.5):
    loc_loss, conf_loss, prob_loss = 0., 0., 0.
    batch_size = labels[0].shape[0]

    for label, pred, anchor, stride in zip(labels, preds, anchors, strides):
        pred_xy = tf.sigmoid(pred[..., :2]) + anchor[..., :2]
        pred_wh = tf.exp(tf.minimum(pred[..., 2:4], tf.math.log(inf))) * anchor[..., 2:]
        # pred_wh = tf.exp(pred[..., 2:4]) * anchor[..., 2:]
        pred_xywh = tf.concat([pred_xy, pred_wh], -1)
        pred_conf = tf.sigmoid(pred[..., 4:5])
        pred_prob = tf.sigmoid(pred[..., 5:])
        
        # _pred_xy = tf.sigmoid(pred[..., :2])
        # _pred_wh = tf.exp(pred[..., :2]) * anchor[..., 2:]
        # _pred_xywh = tf.concat([_pred_xy, _pred_wh], -1)

        # label_xy = label[..., :2]
        # label_wh = label[..., 2:4]
        # label_xywh = tf.concat([label_xy, label_wh], -1)
        label_xywh = label[..., :4]
        resp_mask = label[..., 4:5]
        label_prob = label[..., 5:]
        
        # _label_xy = label[..., :4]
        
        ious = bbox_utils.bbox_iou(pred_xywh, label_xywh)[..., None]

        noresp_mask =  (1.0 - resp_mask) * tf.cast(ious < iou_threshold, dtype=tf.float32)
        
        xy_loss = coord * tf.reduce_sum(resp_mask * tf.square(label_xywh[..., :2] - pred_xywh[..., :2]))
        wh_loss = coord * tf.reduce_sum(resp_mask * tf.square(label_xywh[..., 2:] - pred_xywh[..., 2:]))
        loc_loss += xy_loss + wh_loss
        
        obj_loss = tf.reduce_sum(resp_mask * binaryCrossEntropy(resp_mask, pred_conf, eps))
        noobj_loss = noobj * tf.reduce_sum(noresp_mask * binaryCrossEntropy(resp_mask, pred_conf, eps))
        conf_loss += obj_loss + noobj_loss

        prob_loss += tf.reduce_sum(resp_mask * binaryCrossEntropy(label_prob, pred_prob, eps))
                
    loc_loss /= batch_size
    conf_loss /= batch_size
    prob_loss /= batch_size
    total_loss = loc_loss + conf_loss + prob_loss
    return loc_loss, conf_loss, prob_loss, total_loss

def loss3(labels, preds, anchors, strides, iou_threshold, inf=1e+30, eps=1e-7, coord=5, noobj=0.5):
    loc_loss, conf_loss, prob_loss = 0., 0., 0.
    batch_size = labels[0].shape[0]
    
    for label, pred, anchor, stride in zip(labels, preds, anchors, strides):
        pred_xy = tf.sigmoid(pred[..., :2])
        pred_wh = pred[..., 2:4]
        # pred_xywh = tf.concat([pred_xy, pred_wh], -1)
        pred_conf = tf.sigmoid(pred[..., 4:5])
        pred_prob = tf.sigmoid(pred[..., 5:])
        
        _pred_xy = tf.sigmoid(pred[..., :2]) + anchor[..., :2]
        _pred_wh = tf.exp(pred[..., :2]) * anchor[..., 2:]
        _pred_xywh = tf.concat([_pred_xy, _pred_wh], -1)

        
        label_xy = label[..., :2] - anchor[..., :2]
        label_wh = tf.math.log(tf.maximum(label[..., 2:4]/anchor[..., 2:], eps))
        # label_xywh = tf.concat([label_xy, label_wh], -1)
        resp_mask = label[..., 4:5]
        label_prob = label[..., 5:]
        
        _label_xywh = label[..., :4]
        
        ious = bbox_utils.bbox_iou(_pred_xywh, _label_xywh)[..., None]

        noresp_mask =  (1.0 - resp_mask) * tf.cast(ious < iou_threshold, dtype=tf.float32)
        
        xy_loss = tf.minimum(coord * tf.reduce_sum(resp_mask * tf.square(label_xy - pred_xy)), inf)
        wh_loss = tf.minimum(coord * tf.reduce_sum(resp_mask * tf.square(label_wh - pred_wh)), inf)
        loc_loss += xy_loss + wh_loss
        
        obj_loss = tf.minimum(tf.reduce_sum(resp_mask * binaryCrossEntropy(resp_mask, pred_conf, eps)), inf)
        noobj_loss = tf.minimum(noobj * tf.reduce_sum(noresp_mask * binaryCrossEntropy(resp_mask, pred_conf, eps)), inf)
        conf_loss += obj_loss + noobj_loss

        prob_loss += tf.minimum(tf.reduce_sum(resp_mask * binaryCrossEntropy(label_prob, pred_prob, eps)), inf)
    
    loc_loss /= batch_size
    conf_loss /= batch_size
    prob_loss /= batch_size
    total_loss = loc_loss + conf_loss + prob_loss

    return loc_loss, conf_loss, prob_loss, total_loss