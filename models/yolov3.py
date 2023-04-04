import tensorflow as tf
from tensorflow.keras.layers import Reshape
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from tensorflow.keras import Model
from models.common import *
import numpy as np
from config import *
from utils import anchor_utils, bbox_utils
from losses import yolo_loss

class Darknet53(Layer):
    def __init__(self, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer
        
        self.darkentConv1 = DarknetConv(32, 3, kernel_initializer=self.kernel_initializer)
        self.darknetConv2 = DarknetConv(64, 3, downsample=True, kernel_initializer=self.kernel_initializer)
        
        self.darknetRes1 = [DarknetResidual(64, kernel_initializer=self.kernel_initializer) for _ in range(1)]
        
        self.darknetConv3 = DarknetConv(128, 3, downsample=True, kernel_initializer=self.kernel_initializer)
        
        self.darknetRes2 = [DarknetResidual(128, kernel_initializer=self.kernel_initializer) for _ in range(2)]
        
        self.darknetConv4 = DarknetConv(256, 3, downsample=True,kernel_initializer=self.kernel_initializer)
        
        self.darknetRes3 = [DarknetResidual(256, kernel_initializer=self.kernel_initializer) for _ in range(8)]
        
        self.darknetConv5 = DarknetConv(512, 3, downsample=True, kernel_initializer=self.kernel_initializer)
        
        self.darknetRes4 = [DarknetResidual(512, kernel_initializer=self.kernel_initializer) for _ in range(8)]
        
        self.darknetConv6 = DarknetConv(1024, 3, downsample=True, kernel_initializer=self.kernel_initializer)
        
        self.darknetRes5 = [DarknetResidual(1024, kernel_initializer=self.kernel_initializer) for _ in range(4)]

    def call(self, input, training=False):
        x = self.darkentConv1(input, training)
        x = self.darknetConv2(x, training)

        for i in range(len(self.darknetRes1)):
            x = self.darknetRes1[i](x, training)

        x = self.darknetConv3(x, training)

        for i in range(len(self.darknetRes2)):
            x = self.darknetRes2[i](x, training)

        x = self.darknetConv4(x, training)

        for i in range(len(self.darknetRes3)):
            x = self.darknetRes3[i](x, training)
        
        s_route = x
        x = self.darknetConv5(x, training)

        for i in range(len(self.darknetRes4)):
            x = self.darknetRes4[i](x, training)
        
        m_route = x
        x = self.darknetConv6(x, training)

        for i in range(len(self.darknetRes5)):
            x = self.darknetRes5[i](x, training)

        return s_route, m_route, x

class Model(Model):
    def __init__(self, anchors=ANCHORS, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, strides=STRIDES,
                 iou_threshold=IOU_THRESHOLD, num_anchor=NUM_ANCHORS, eps=EPS, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.strides = np.array(strides)
        self.anchors = anchor_utils.get_anchors_xywh(anchors, strides, image_size)
        self.scales = image_size//self.strides
        self.iou_threshold = iou_threshold
        self.num_anchor = num_anchor
        self.eps = eps
        self.inf = 1e+30
        self.kernel_initializer = kernel_initializer

        self.darknet53 = Darknet53()
        
        self.large_layers  = [DarknetConv(512, 1, kernel_initializer=self.kernel_initializer),
                              DarknetConv(1024, 3, kernel_initializer=self.kernel_initializer),
                              DarknetConv(512, 1, kernel_initializer=self.kernel_initializer),
                              DarknetConv(1024, 3, kernel_initializer=self.kernel_initializer),
                              DarknetConv(512, 1, kernel_initializer=self.kernel_initializer)]
        
        self.large_branch_layers = [DarknetConv(1024, 3, kernel_initializer=self.kernel_initializer),
                             DarknetConv(self.num_anchor*(5+self.num_classes), 1, activate=False, bn=False, kernel_initializer=self.kernel_initializer)]
        
        self.large_upsample_layers = [DarknetConv(256, 1, kernel_initializer=self.kernel_initializer),
                               DarknetUpsample()]
        
        self.medium_layers  = [DarknetConv(256, 1, kernel_initializer=self.kernel_initializer),
                               DarknetConv(512, 3, kernel_initializer=self.kernel_initializer),
                               DarknetConv(256, 1, kernel_initializer=self.kernel_initializer),
                               DarknetConv(512, 3, kernel_initializer=self.kernel_initializer),
                               DarknetConv(256, 1, kernel_initializer=self.kernel_initializer)]
        
        self.medium_branch_layers = [DarknetConv(512, 3, kernel_initializer=self.kernel_initializer),
                                     DarknetConv(self.num_anchor*(5+self.num_classes), 1, activate=False, bn=False, kernel_initializer=self.kernel_initializer)]
        
        self.medium_upsample_layers = [DarknetConv(128, 1, kernel_initializer=self.kernel_initializer),
                                       DarknetUpsample()]
        
        self.small_layers = [DarknetConv(128, 1, kernel_initializer=self.kernel_initializer),
                             DarknetConv(256, 3, kernel_initializer=self.kernel_initializer),
                             DarknetConv(128, 1, kernel_initializer=self.kernel_initializer),
                             DarknetConv(256, 3, kernel_initializer=self.kernel_initializer),
                             DarknetConv(128, 1, kernel_initializer=self.kernel_initializer)]
        
        self.small_branch_layers = [DarknetConv(256, 3, kernel_initializer=self.kernel_initializer),
                                    DarknetConv(self.num_anchor*(5+self.num_classes), 1, activate=False, bn=False, kernel_initializer=self.kernel_initializer)]

        print('Model: YOLOv3')
    
    def call(self, input, training=False):
        s_route, m_route, l_route = self.darknet53(input, training)
        
        for i in range(len(self.large_layers)):
            l_route = self.large_layers[i](l_route, training)
        
        large_branch = l_route
        for i in range(len(self.large_branch_layers)):
            large_branch = self.large_branch_layers[i](large_branch, training)
        
        lbbox = Reshape((self.scales[2], self.scales[2], self.num_anchor, 5+self.num_classes))(large_branch)

        # mbbox
        for i in range(len(self.large_upsample_layers)):
            l_route = self.large_upsample_layers[i](l_route, training)
        
        m_route = tf.concat([l_route, m_route], -1)
        for i in range(len(self.medium_layers)):
            m_route = self.medium_layers[i](m_route, training)
        
        medium_branch = m_route
        for i in range(len(self.medium_branch_layers)):
            medium_branch = self.medium_branch_layers[i](medium_branch, training)
        
        mbbox = Reshape((self.scales[1], self.scales[1], self.num_anchor, 5+self.num_classes))(medium_branch)
        
        # sbbox
        for i in range(len(self.medium_upsample_layers)):
            m_route = self.medium_upsample_layers[i](m_route, training)
        
        s_route = tf.concat([m_route, s_route], -1)
        for i in range(len(self.small_layers)):
            s_route = self.small_layers[i](s_route, training)
        
        small_branch = s_route
        for i in range(len(self.small_branch_layers)):
            small_branch = self.small_branch_layers[i](small_branch, training)
        
        sbbox = Reshape((self.scales[0], self.scales[0], self.num_anchor, 5+self.num_classes))(small_branch)

        return sbbox, mbbox, lbbox

    def loss(self, labels, preds):
        return yolo_loss.loss3(labels, preds, self.anchors, self.strides, self.iou_threshold, self.inf, self.eps)