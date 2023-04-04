import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, MaxPooling2D
from tensorflow.keras import Model
from models.common import *
import numpy as np
from config import *
from utils import anchor_utils, bbox_utils
from losses import yolo_loss

class Darknet19_tiny(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.darknetConv1 = DarknetConv(16, 3)
        
        self.darknetConv2 = DarknetConv(32, 3)
        
        self.darknetConv3 = DarknetConv(64, 3)

        self.darknetConv4 = DarknetConv(128, 3)
        
        self.darknetConv5 = DarknetConv(256, 3)
        
        self.darknetConv6 = DarknetConv(512, 3)

        self.darknetConv7 = DarknetConv(1024, 3)

    def call(self, input, training=False):
        x = self.darknetConv1(input, training)
        x = MaxPooling2D(2, 2)(x)

        x = self.darknetConv2(x, training)
        x = MaxPooling2D(2, 2)(x)

        x = self.darknetConv3(x, training)
        x = MaxPooling2D(2, 2)(x)

        x = self.darknetConv4(x, training)
        x = MaxPooling2D(2, 2)(x)

        x = self.darknetConv5(x, training)
        m_route = x
        x = MaxPooling2D(2, 2)(x)

        x = self.darknetConv6(x, training)
        x = MaxPooling2D(2, 1, 'same')(x)

        l_route = self.darknetConv7(x, training)
        
        return m_route, l_route
    
class Model(Model):
    def __init__(self, anchors=ANCHORS, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, strides=STRIDES,
                 coord=5, noobj=0.5, iou_threshold=IOU_THRESHOLD, num_anchor=NUM_ANCHORS, eps=EPS, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.strides = np.array(strides)
        self.anchors = anchor_utils.get_anchors_xywh(anchors, strides, image_size)
        self.scales = image_size//self.strides
        self.coord = coord
        self.noobj = noobj
        self.iou_threshold = iou_threshold
        self.num_anchor = num_anchor
        self.eps = eps
        self.inf = 1e+30

        self.darknet19_tiny = Darknet19_tiny()
        
        self.darknetConv8 = DarknetConv(256, 1)
        
        self.large_branch_layers = [DarknetConv(512, 3),
                                    DarknetConv(self.num_anchor*(5 + self.num_classes), 1, activate=False, bn=False)]
        
        self.large_upsample_layers = [DarknetConv(128, 1),
                                      DarknetUpsample()]
        
        self.medium_branch_layers = [DarknetConv(256, 3),
                                     DarknetConv(self.num_anchor*(5 + self.num_classes), 1, activate=False, bn=False)]
        print('Model: YOLOv3_tiny')
    def call(self, input, training):
        m_route, l_route = self.darknet19_tiny(input, training)
        
        l_route = self.darknetConv8(l_route, training)
        
        large_branch = l_route
        for i in range(len(self.large_branch_layers)):
            large_branch = self.large_branch_layers[i](large_branch, training)
            
        lbbox = Reshape((self.scales[1], self.scales[1], self.num_anchor, 5 + self.num_classes))(large_branch)
        
        for i in range(len(self.large_upsample_layers)):
            l_route = self.large_upsample_layers[i](l_route, training)
            
        m_route = tf.concat([m_route, l_route], -1)
        medium_branch = m_route
        for i in range(len(self.medium_branch_layers)):
            medium_branch = self.medium_branch_layers[i](medium_branch, training)
        
        mbbox = Reshape((self.scales[0], self.scales[0], self.num_anchor, 5 + self.num_classes))(medium_branch)
        
        return mbbox, lbbox
    
    def loss(self, labels, preds):
        return yolo_loss.loss3(labels, preds, self.anchors, self.strides, self.iou_threshold, self.inf, self.eps)