import tensorflow as tf
from losses.common import *

@tf.function
def v3_prob_loss(label_cprob, pred_prob_raw, inf, eps):
    resp_mask = label_cprob[..., :1]
    label_prob = label_cprob[..., 1:]
    
    pred_prob = tf.sigmoid(pred_prob_raw)
    
    prob_loss = tf.reduce_sum(tf.minimum(resp_mask * binaryCrossEntropy(label_prob, pred_prob, eps), inf), [1,2,3,4])
    
    return prob_loss

def v2_prob_loss(label_cprob, pred_prob_raw, inf, eps):
    resp_mask = label_cprob[..., :1]
    label_prob = label_cprob[..., 1:]
    
    pred_prob = tf.nn.softmax(pred_prob_raw, -1)
    
    prob_loss = tf.reduce_sum(tf.minimum(resp_mask * tf.square(label_prob - pred_prob), inf), [1,2,3,4])
    
    return prob_loss