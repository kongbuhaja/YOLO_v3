import tensorflow as tf

@tf.function
def v3_loc_loss(label_xywhc, pred_xywh_raw, anchor, inf, eps, coord):
    pred_xy = tf.sigmoid(pred_xywh_raw[..., :2])
    pred_wh = pred_xywh_raw[..., 2:4]
    
    label_xywh = label_xywhc[..., :4]
    resp_mask = label_xywhc[..., 4:]
    
    label_xy = label_xywh[..., :2] - anchor[..., :2]
    label_wh = tf.math.log(tf.maximum(label_xywh[..., 2:]/anchor[..., 2:], eps))
    
    xy_loss = coord * tf.reduce_sum(tf.minimum(resp_mask * tf.square(label_xy - pred_xy), inf), [1,2,3,4])
    wh_loss = coord * tf.reduce_sum(tf.minimum(resp_mask * tf.square(label_wh - pred_wh), inf), [1,2,3,4])
    
    return xy_loss + wh_loss

@tf.function
def v3_paper_loc_loss(label_xywhc, pred_xywh_raw, anchor, inf, eps, coord):
    pred_xy = pred_xywh_raw[..., :2]
    pred_wh = pred_xywh_raw[..., 2:]
    
    label_xywh = label_xywhc[..., :4]
    resp_mask = label_xywhc[..., 4:5]
    
    sig_xy = tf.maximum(label_xywh[..., :2] - anchor[..., :2], eps)
    label_xy = tf.math.log(sig_xy) - tf.math.log(1-sig_xy)
    label_wh = tf.math.log(tf.maximum(label_xywh[..., 2:] / anchor[..., 2:], eps))
    
    # xy_loss = tf.minimum(coord * tf.reduce_sum(resp_mask * tf.square(label_xy - pred_xy)), inf)
    # wh_loss = tf.minimum(coord * tf.reduce_sum(resp_mask * tf.square(label_wh - pred_wh)), inf)
    
    xy_loss = coord * tf.reduce_sum(tf.minimum(resp_mask * tf.abs(label_xy - pred_xy), inf), [1,2,3,4])
    wh_loss = coord * tf.reduce_sum(tf.minimum(resp_mask * tf.abs(label_wh - pred_wh), inf), [1,2,3,4])
    
    return xy_loss + wh_loss

@tf.function    
def v2_loc_loss(label_xywhc, pred_xywh_raw, anchor, inf, eps, coord):
    pred_xy = tf.sigmoid(pred_xywh_raw[..., :2]) + anchor[..., :2]
    pred_wh = tf.exp(tf.minimum(pred_xywh_raw[..., 2:], tf.math.log(inf))) * anchor[..., 2:]
    
    label_xy = label_xywhc[..., :2]
    label_wh = label_xywhc[..., 2:4]
    resp_mask = label_xywhc[..., 4:5]
    
    xy_loss = coord * tf.reduce_sum(tf.minimum(resp_mask * tf.square(label_xy - pred_xy), inf), [1,2,3,4])
    wh_loss = coord * tf.reduce_sum(tf.minimum(resp_mask * tf.square(label_wh - pred_wh), inf), [1,2,3,4])
    
    return xy_loss + wh_loss
    