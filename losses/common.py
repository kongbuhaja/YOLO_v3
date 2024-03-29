import tensorflow as tf

@tf.function
def binaryCrossEntropy(label, pred, eps=1e-7):
    pred = tf.minimum(tf.maximum(pred, eps), 1-eps)
    return -(label*tf.math.log(pred) + (1.-label)*tf.math.log(1.-pred))