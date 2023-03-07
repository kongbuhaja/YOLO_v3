import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras import Model
from models.common import *
import numpy as np
from config import *
from utils import anchor_utils, bbox_utils

class Darknet53(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.darkentConv1 = DarknetConv(32, 3)
        self.darknetConv2 = DarknetConv(64, 3, downsample=True)
        
        self.darknetRes1 = [DarknetResidual(64) for _ in range(1)]
        
        self.darknetConv3 = DarknetConv(128, 3, downsample=True)
        
        self.darknetRes2 = [DarknetResidual(128) for _ in range(2)]
        
        self.darknetConv4 = DarknetConv(256, 3, downsample=True)
        
        self.darknetRes3 = [DarknetResidual(256) for _ in range(8)]
        
        self.darknetConv5 = DarknetConv(512, 3, downsample=True)
        
        self.darknetRes4 = [DarknetResidual(512) for _ in range(8)]
        
        self.darknetConv6 = DarknetConv(1024, 3, downsample=True)
        
        self.darknetRes5 = [DarknetResidual(1024) for _ in range(4)]

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
        self.inf = 30.

        self.darknet53 = Darknet53()
        
        self.large_layers  = [DarknetConv(512, 1),
                              DarknetConv(1024, 3),
                              DarknetConv(512, 1),
                              DarknetConv(1024, 3),
                              DarknetConv(512, 1)]
        
        self.large_branch_layers = [DarknetConv(1024, 3),
                             DarknetConv(self.num_anchor*(5+self.num_classes), 1, activate=False, bn=False)]
        
        self.large_upsample_layers = [DarknetConv(256, 1),
                               DarknetUpsample()]
        
        self.medium_layers  = [DarknetConv(256, 1),
                               DarknetConv(512, 3),
                               DarknetConv(256, 1),
                               DarknetConv(512, 3),
                               DarknetConv(256, 1)]
        
        self.medium_branch_layers = [DarknetConv(512, 3),
                                     DarknetConv(self.num_anchor*(5+self.num_classes), 1, activate=False, bn=False)]
        
        self.medium_upsample_layers = [DarknetConv(128, 1),
                                       DarknetUpsample()]
        
        self.small_layers = [DarknetConv(128, 1),
                             DarknetConv(256, 3),
                             DarknetConv(128, 1),
                             DarknetConv(256, 3),
                             DarknetConv(128, 1)]
        
        self.small_branch_layers = [DarknetConv(256, 3),
                                    DarknetConv(self.num_anchor*(5+self.num_classes), 1, activate=False, bn=False)]

        print('Model: YOLOv3')
    
    def call(self, input, training=False):
        s_route, m_route, l_route = self.darknet53(input, training)
        
        for i in range(len(self.large_layers)):
            l_route = self.large_layers[i](l_route, training)
        
        large_branch = l_route
        for i in range(len(self.large_branch_layers)):
            large_branch = self.large_branch_layers[i](large_branch, training)
        
        lbbox = Reshape((self.scales[2], self.scales[2], 3, 5+self.num_classes))(large_branch)

        for i in range(len(self.medium_layers)):
            m_route = self.medium_layers[i](m_route, training)
        
        medium_branch = m_route
        for i in range(len(self.medium_branch_layers)):
            medium_branch = self.medium_branch_layers[i](medium_branch, training)
        
        mbbox = Reshape((self.scales[1], self.scales[1], 3, 5+self.num_classes))(medium_branch)
        
        for i in range(len(self.small_layers)):
            s_route = self.small_layers[i](s_route, training)
        
        small_branch = s_route
        for i in range(len(self.small_branch_layers)):
            small_branch = self.small_branch_layers[i](small_branch, training)
        
        sbbox = Reshape((self.scales[0], self.scales[0], 3, 5+self.num_classes))(small_branch)

        return sbbox, mbbox, lbbox

    def binaryCrossEntropy(self, label, pred):
        pred = tf.minimum(tf.maximum(pred, self.eps), 1-self.eps)
        return -(label*tf.math.log(pred) + (1.-label)*tf.math.log(1.-pred))

    def loss(self, labels, preds):
        loc_loss, conf_loss, prob_loss = 0., 0., 0.
        batch_size = labels[0].shape[0]

        for label, pred, anchor, stride in zip(labels, preds, self.anchors, self.strides):
            pred_xy = pred[..., :2]
            pred_wh = pred[..., 2:4]
            pred_conf = tf.sigmoid(pred[..., 4:5])
            pred_prob = tf.sigmoid(pred[..., 5:])

            label_xywh_ = label[..., :4]
            resp_mask = label[..., 4:5]
            label_prob = label[..., 5:]
            
            sig_xy = tf.maximum(label[..., :2] - anchor[..., :2] , self.eps)
            label_xy = tf.math.log(sig_xy) - tf.math.log(1-sig_xy)
            label_wh = tf.math.log(tf.maximum(label[..., 2:4] / anchor[..., 2:], self.eps))            

            pred_xy_ = tf.sigmoid(pred[..., :2]) + anchor[..., :2]
            pred_wh_ = tf.exp(tf.minimum(pred[..., 2:4], self.inf)) * anchor[..., 2:]
            pred_xywh_ = tf.concat([pred_xy_, pred_wh_], -1)
            
            ious = tf.expand_dims(bbox_utils.bbox_iou(pred_xywh_, label_xywh_), -1)

            noresp_mask =  (1.0 - resp_mask) * tf.cast(ious < self.iou_threshold, dtype=tf.float32)
            
            xy_loss = self.coord * tf.reduce_sum(resp_mask * tf.square(label_xy - pred_xy))
            wh_loss = self.coord * tf.reduce_sum(resp_mask * tf.square(label_wh - pred_wh))
            loc_loss += xy_loss + wh_loss
            
            obj_loss = tf.reduce_sum(resp_mask * self.binaryCrossEntropy(resp_mask, pred_conf))
            noobj_loss = self.noobj * tf.reduce_sum(noresp_mask * self.binaryCrossEntropy(resp_mask, pred_conf))
            conf_loss += obj_loss + noobj_loss

            prob_loss += 5*tf.reduce_sum(resp_mask * self.binaryCrossEntropy(label_prob, pred_prob))
            
        loc_loss /= batch_size
        conf_loss /= batch_size
        prob_loss /= batch_size
        total_loss = loc_loss + conf_loss + prob_loss
        return loc_loss, conf_loss, prob_loss, total_loss
    
    def loss2(self, labels, preds):
        loc_loss, conf_loss, prob_loss = 0., 0., 0.
        batch_size = labels[0].shape[0]

        for label, pred, anchor, stride in zip(labels, preds, self.anchors, self.strides):
            pred_xy = tf.sigmoid(pred[..., :2]) + anchor[..., :2]
            pred_wh = tf.exp(tf.minimum(pred[..., 2:4], self.inf)) * anchor[..., 2:]
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

            noresp_mask =  (1.0 - resp_mask) * tf.cast(ious < self.iou_threshold, dtype=tf.float32)
            
            xy_loss = self.coord * tf.reduce_sum(resp_mask * tf.square(label_xywh[..., :2] - pred_xywh[..., :2]))
            wh_loss = self.coord * tf.reduce_sum(resp_mask * tf.square(label_xywh[..., 2:] - pred_xywh[..., 2:]))
            loc_loss += xy_loss + wh_loss
            
            obj_loss = tf.reduce_sum(resp_mask * self.binaryCrossEntropy(resp_mask, pred_conf))
            noobj_loss = self.noobj * tf.reduce_sum(noresp_mask * self.binaryCrossEntropy(resp_mask, pred_conf))
            conf_loss += obj_loss + noobj_loss

            prob_loss += tf.reduce_sum(resp_mask * self.binaryCrossEntropy(label_prob, pred_prob))
                    
        loc_loss /= batch_size
        conf_loss /= batch_size
        prob_loss /= batch_size
        total_loss = loc_loss + conf_loss + prob_loss
        return loc_loss, conf_loss, prob_loss, total_loss
    
    def loss3(self, labels, preds):
        loc_loss, conf_loss, prob_loss = 0., 0., 0.
        batch_size = labels[0].shape[0]

        for label, pred, anchor, stride in zip(labels, preds, self.anchors, self.strides):
            pred_xy = tf.sigmoid(pred[..., :2])
            pred_wh = pred[..., 2:4]
            # pred_xywh = tf.concat([pred_xy, pred_wh], -1)
            pred_conf = tf.sigmoid(pred[..., 4:5])
            pred_prob = tf.sigmoid(pred[..., 5:])
            
            _pred_xy = tf.sigmoid(pred[..., :2]) + anchor[..., :2]
            _pred_wh = tf.exp(pred[..., :2]) * anchor[..., 2:]
            _pred_xywh = tf.concat([_pred_xy, _pred_wh], -1)
  
            
            label_xy = label[..., :2] - anchor[..., :2]
            label_wh = tf.math.log(tf.maximum(label[..., 2:4]/anchor[..., 2:], self.eps))
            # label_xywh = tf.concat([label_xy, label_wh], -1)
            resp_mask = label[..., 4:5]
            label_prob = label[..., 5:]
            
            _label_xywh = label[..., :4]
            
            ious = bbox_utils.bbox_iou(_pred_xywh, _label_xywh)[..., None]

            noresp_mask =  (1.0 - resp_mask) * tf.cast(ious < self.iou_threshold, dtype=tf.float32)
            
            xy_loss = tf.minimum(self.coord * tf.reduce_sum(resp_mask * tf.square(label_xy - pred_xy)), 1e+30)
            wh_loss = tf.minimum(self.coord * tf.reduce_sum(resp_mask * tf.square(label_wh - pred_wh)), 1e+30)
            loc_loss += xy_loss + wh_loss
            
            obj_loss = tf.minimum(tf.reduce_sum(resp_mask * self.binaryCrossEntropy(resp_mask, pred_conf)), 1e+30)
            noobj_loss = tf.minimum(self.noobj * tf.reduce_sum(noresp_mask * self.binaryCrossEntropy(resp_mask, pred_conf)), 1e+30)
            conf_loss += obj_loss + noobj_loss

            prob_loss += tf.minimum(tf.reduce_sum(resp_mask * self.binaryCrossEntropy(label_prob, pred_prob)), 1e+30)
        
        loc_loss /= batch_size
        conf_loss /= batch_size
        prob_loss /= batch_size
        total_loss = loc_loss + conf_loss + prob_loss
        return loc_loss, conf_loss, prob_loss, total_loss

# def Create_Yolo(input_size=416, channels=3, training=False, CLASSES=YOLO_COCO_CLASSES):
#     NUM_CLASS = len(read_class_names(CLASSES))
#     input_layer  = Input([input_size, input_size, channels])

#     if TRAIN_YOLO_TINY:
#         conv_tensors = YOLOv3_tiny(input_layer, NUM_CLASS)
#     else:
#         conv_tensors = YOLOv3(input_layer, NUM_CLASS)

#     output_tensors = []
#     for i, conv_tensor in enumerate(conv_tensors):
#         pred_tensor = decode(conv_tensor, NUM_CLASS, i)
#         if training: output_tensors.append(conv_tensor)
#         output_tensors.append(pred_tensor)

#     Yolo = tf.keras.Model(input_layer, output_tensors)
#     return Yolo

# class Yolov3Loss():
#     def __init__(self, anchors, image_size, strides, iou_threshold, coord=5, noobj=0.5):
#         self.anchors = anchor_utils.get_anchors_xywh(anchors, image_size, strides)
#         self.image_size = image_size
#         self.strides = strides
#         self.iou_threshold = iou_threshold
#         self.coord = coord
#         self.noobj = 0.5

#     def binaryCrossEtropy(label, pred):
#         return -(label*tf.math.log(pred) + (1.-label)*tf.math.log(1.-pred))
 
#     def loss(self, pred_grids, labels):
#         loc_loss, conf_loss, prob_loss = 0, 0, 0
        
#         for pred_grid, label, anchor, stride in zip(pred_grids, labels, self.anchors, self.strides):
#             batch_size = label.shape[:2]

#             pred_xy = tf.sigmoid(pred_grid[...,  :2]) + anchor[:2]
#             pred_wh = tf.exp(pred_grid[..., 2:4]) * anchor[2:]
#             pred_xywh = tf.concat([pred_xy, pred_wh], -1) * stride
#             pred_conf = tf.sigmoid(pred_grid[..., 4:5])
#             pred_prob = tf.sigmoid(pred_grid[..., 5: ])

#             label_xywh = label[..., :4]
#             resp_mask = label[..., 4:5]
#             label_prob = label[..., 5:]

#             iou = tf.expand_dims(bbox_utils.bbox_iou_xywh(pred_xywh, label_xywh), -1)
#             max_iou = tf.reduce_max(iou, -1)
#             noresp_mask = (1.0 - resp_mask) * tf.cast(max_iou < self.iou_threshold, tf.float32)

#             xy_loss = self.coord * tf.reduce_sum(resp_mask * (label_xywh[..., :2] - pred_xywh[..., :2])**2)
#             wh_loss = self.coord * tf.reduce_sum(resp_mask * (tf.sqrt(label_xywh[..., 2:]) - tf.sqrt(pred_xywh[..., 2:]))**2)
#             loc_loss += xy_loss + wh_loss
            
#             conf_loss += tf.reduce_sum(resp_mask * self.binaryCrossEtropy(resp_mask, pred_conf) + 
#                                        self.noobj * noresp_mask * self.binaryCrossEtropy(resp_mask, pred_conf))

#             prob_loss += tf.reduce_sum(resp_mask * self.binaryCrossEtropy(label_prob, pred_prob))       

#         return (loc_loss + conf_loss + prob_loss)/batch_size
