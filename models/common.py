import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Layer, LeakyReLU, Add
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

class DarknetBatchN(BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

class DarknetConv(Layer):
    def __init__(self, units, kernel_size, kernel_initializer=glorot, downsample=False, activate=True, bn=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.downsample = downsample
        self.activate = activate
        self.bn = bn
        self.padding = 'same'
        self.strides = 2 if self.downsample else 1
        
        self.conv = Conv2D(self.units, self.kernel_size, padding=self.padding, strides=self.strides,
                           use_bias=not self.bn, kernel_regularizer=l2(0.0005),
                           kernel_initializer=self.kernel_initializer)
        self.batchN = BatchNormalization()
    def call(self, input, training=False):
        # x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(input)
        conv = self.conv(input)
        if self.bn:
            conv = self.batchN(conv, training)
        if self.activate:
            conv = LeakyReLU(alpha=0.1)(conv)
        
        return conv

class DarknetResidual(Layer):
    def __init__(self, units, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        
        self.conv1 = DarknetConv(self.units//2, 1, kernel_initializer=self.kernel_initializer)
        self.conv2 = DarknetConv(self.units, 3, kernel_initializer=self.kernel_initializer)
    
    def call(self, input, training=False):
        short_cut = input
        conv = self.conv1(input, training)
        conv = self.conv2(conv, training)

        return Add()([short_cut, conv])
    
class DarknetUpsample(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input, training=False):
        return tf.image.resize(input, (input.shape[1]*2, input.shape[2]*2), method='nearest')
