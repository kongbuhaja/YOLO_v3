import tensorflow as tf
from config import *
import numpy as np
from utils import aug_utils, bbox_utils, anchor_utils

class DataLoader():
    def __init__(self, dtype=DTYPE, batch_size=BATCH_SIZE, anchors=ANCHORS, num_classes=NUM_CLASSES,
                 image_size=IMAGE_SIZE, strides=STRIDES, iou_threshold=IOU_THRESHOLD, max_bboxes=MAX_BBOXES):
                 
        self.batch_size = batch_size
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.image_size = image_size
        self.strides = np.array(strides)
        self.anchors = np.array(anchors)
        self.iou_threshold = iou_threshold
        self.max_bboxes = max_bboxes
        self.dtype = dtype

    def __call__(self, split, use_tfrecord=True, use_label=False):
        if self.dtype == 'voc':
            from datasets.voc_dataset import Dataset
        elif self.dtype == 'custom':
            from datasets.custom_dataset import Dataset
        dataset = Dataset(split)

        if use_tfrecord:
            dataset.make_tfrecord()
            data = dataset.read_tfrecord()
            
        else:
            dataset.load()
            data_gen = dataset.generator
            data = tf.data.Dataset.from_generator(data_gen, (tf.uint8, tf.float32))

        data = data.cache()
        
        if split == 'train':
            data = data.shuffle(buffer_size = self.length(split) * 10)
            # data = data.map(self.py_augmentation, num_parallel_calls=-1)
            data = data.map(aug_utils.tf_augmentation, num_parallel_calls=-1)
        
        # data = data.map(self.py_preprocessing, num_parallel_calls=-1)
        data = data.map(self.tf_preprocessing, num_parallel_calls=-1)
        data = data.padded_batch(self.batch_size, padded_shapes=get_padded_shapes(), padding_values=get_padding_values(), drop_remainder=True)
        
        if use_label:
            data = data.prefetch(1)
        else:
            # data = data.map(self.tf_labels_to_grids, num_parallel_calls=-1).prefetch(1)
            data = data.map(self.py_labels_to_grids, num_parallel_calls=-1).prefetch(1)
        return data
    
    def length(self, split):
        infopath = f'./data/{self.dtype}/{split}.txt'
        with open(infopath, 'r') as f:
            lines = f.readlines()
        return int(lines[0])
    
    def py_augmentation(self, image, labels, width, height):
        new_image, labels, width, height = tf.py_function(aug_utils.augmentation, [image, labels, width, height], [tf.uint8, tf.float32, tf.float32, tf.float32])
        return new_image, labels, width, height
    
    def py_preprocessing(self, image, labels, width, height):
        image, labels = tf.py_function(self.preprocessing, [image, labels], [tf.uint8, tf.float32])
        return tf.cast(image, tf.float32), labels
    
    def preprocessing(self, image, labels):
        image, labels = aug_utils.resize_padding(np.array(image), np.array(labels), self.image_size)
        labels = bbox_utils.xyxy_to_xywh(labels, True)
        
        return image, labels
    
    def py_labels_to_grids(self, image, labels):
        s_grid, m_grid, l_grid = tf.py_function(self.labels_to_grids, [labels], [tf.float32, tf.float32, tf.float32])
        return image, s_grid, m_grid, l_grid
    
    @tf.function
    def tf_preprocessing(self, image, labels, width, height):
        image, labels = aug_utils.tf_resize_padding(image, labels, width, height, self.image_size)
        labels = bbox_utils.xyxy_to_xywh(labels, True)
        return tf.cast(image, tf.float32)/255., labels
    
    @tf.function
    def tf_labels_to_grids(self, image, labels):
        s_grid, m_grid, l_grid = self.labels_to_grids2(labels)
        return tf.cast(image, tf.float32), s_grid, m_grid, l_grid
        
    def labels_to_grids(self, labels):
        grids = []
        best_ious = tf.zeros((self.batch_size, self.max_bboxes))
        anchors = anchor_utils.get_anchors_xywh(self.anchors, self.strides, self.image_size)
    
        no_obj = tf.reduce_sum(labels[..., 2:4], -1) == 0
        conf = tf.cast(tf.where(no_obj, tf.zeros_like(no_obj), tf.ones_like(no_obj)), tf.float32)[..., None]
        onehot = tf.where(no_obj[..., None], tf.zeros_like(conf), tf.one_hot(tf.cast(labels[..., 4], dtype=tf.int32), NUM_CLASSES))
        conf_onehot = tf.concat([conf, onehot], -1)
        
        for i in range(self.strides.shape[0]):
            anchor = tf.concat([anchors[i][..., :2] + 0.5, anchors[i][..., 2:]],-1)
            scaled_bboxes = labels[..., :4] / self.strides[i]
            
            ious = bbox_utils.bbox_iou(anchor[..., None,:], scaled_bboxes[:,None,None,None])
            
            new_labels =  tf.concat([scaled_bboxes, conf_onehot], -1)
            
            max_ious_id = tf.argmax(ious, -1)
            max_ious_mask = tf.cast(tf.reduce_max(ious, -1)[..., None] >= self.iou_threshold, tf.float32)
            grid = tf.gather(new_labels, max_ious_id, batch_dims=1) * max_ious_mask
            grids.append(grid)
            
            best_ious = tf.maximum(tf.reduce_max(ious, [1,2,3]), best_ious)

        if tf.reduce_any(best_ious < self.iou_threshold):
            for i in range(self.strides.shape[0]):
                anchor = tf.concat([anchors[i][..., :2] + 0.5, anchors[i][..., 2:]],-1)
                
                scaled_bboxes = labels[..., :4] / self.strides[i]
                
                ious = bbox_utils.bbox_iou(anchor[..., None,:], scaled_bboxes[:,None,None,None])
                non_zero_ious_mask = tf.cast(tf.where(best_ious!=0, tf.ones_like(best_ious), tf.zeros_like(best_ious)), tf.bool)[:,None,None,None]

                best_mask = tf.reduce_any(tf.math.logical_and((ious == best_ious[:,None,None,None]), non_zero_ious_mask), -1)[..., None]
                
                if tf.reduce_any(best_mask):
                    new_labels =  tf.concat([scaled_bboxes, conf_onehot], -1)
                    
                    max_ious_id = tf.argmax(ious, -1)
                    best_masked_grid = tf.gather(new_labels, max_ious_id, batch_dims=1) * tf.cast(best_mask, tf.float32)
                    
                    grids[i] = tf.where(best_mask, best_masked_grid, grids[i])
        return grids
    
    @tf.function
    def labels_to_grids2(self, labels):
        grids = [tf.zeros((self.batch_size, self.image_size//stride, self.image_size//stride , self.num_anchors, 5+self.num_classes)) for stride in self.strides]
        best_ious = tf.zeros((self.batch_size, self.max_bboxes))
        best_mask = [tf.zeros((self.batch_size, self.image_size//stride, self.image_size//stride , self.num_anchors, 1), dtype=tf.bool) for stride in self.strides]
        ious = [tf.zeros((self.batch_size, self.image_size//stride, self.image_size//stride , self.num_anchors, 100), dtype=tf.bool) for stride in self.strides]
        anchors = anchor_utils.get_anchors_xywh(self.anchors, self.strides, self.image_size)
    
        no_obj = tf.reduce_sum(labels[..., 2:4], -1) == 0
        conf = tf.cast(tf.where(no_obj, tf.zeros_like(no_obj), tf.ones_like(no_obj)), tf.float32)[..., None]
        onehot = tf.where(no_obj[..., None], tf.zeros_like(conf), tf.one_hot(tf.cast(labels[..., 4], dtype=tf.int32), NUM_CLASSES))
        conf_onehot = tf.concat([conf, onehot], -1)
        
        for i in range(self.num_anchors):
            anchor = tf.concat([anchors[i][..., :2] + 0.5, anchors[i][..., 2:]],-1)
            scaled_bboxes = labels[..., :4] / self.strides[i]
            
            ious[i] = bbox_utils.bbox_iou(anchor[..., None,:], scaled_bboxes[:,None,None,None])

            new_labels =  tf.concat([scaled_bboxes, conf_onehot], -1)
            
            max_ious_id = tf.argmax(ious[i], -1)
            max_ious_mask = tf.cast(tf.reduce_max(ious[i], -1)[..., None] >= self.iou_threshold, tf.float32)
            grid = tf.gather(new_labels, max_ious_id, batch_dims=1) * max_ious_mask
            grids[i] = grid
            
            best_ious = tf.maximum(tf.reduce_max(ious[i], [1,2,3]), best_ious)
        
        if tf.reduce_any(best_ious < self.iou_threshold):
            for i in range(self.num_anchors):
                anchor = tf.concat([anchors[i][..., :2] + 0.5, anchors[i][..., 2:]],-1)
                
                scaled_bboxes = labels[..., :4] / self.strides[i]
                
                # ious = bbox_utils.bbox_iou(anchor[..., None,:], scaled_bboxes[:,None,None,None])
                non_zero_ious_mask = tf.cast(tf.where(best_ious!=0, tf.ones_like(best_ious), tf.zeros_like(best_ious)), tf.bool)[:,None,None,None]
                # print(best_ious[:,None,None,None].shape, ious.shape, non_zero_ious_mask.shape)
                best_mask[i] = tf.reduce_any(tf.math.logical_and((ious[i] == best_ious[:,None,None,None]), non_zero_ious_mask), -1)[..., None]
                
                if tf.reduce_any(best_mask[i]):
                    new_labels =  tf.concat([scaled_bboxes, conf_onehot], -1)
                    print(best_mask[i].dtype)
                    max_ious_id = tf.argmax(ious[i], -1)
                    best_masked_grid = tf.gather(new_labels, max_ious_id, batch_dims=1) * tf.cast(best_mask[i], tf.float32)
                
                    grids[i] = tf.where(best_mask[i], best_masked_grid, grids[i])
        return grids
    
def get_padded_shapes():
    return [None, None, None], [MAX_BBOXES, None]

# def get_batch_padded_shapes():
    

def get_padding_values():
    return tf.constant(0, tf.float32), tf.constant(0, tf.float32)