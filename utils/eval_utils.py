import tensorflow as tf
import numpy as np
from utils import bbox_utils
from config import *

class stats:
    def __init__(self, labels=LABELS, iou_threshold=IOU_THRESHOLD):
        self.stats = {}
        for i, label in enumerate(labels):
            self.stats[i] = {
                'label': label,
                'total': 0,
                'tp': [],
                'fp': [],
                'scores':[],
            }
        self.iou_threshold = iou_threshold
        self.mAP = 0.0

    def update_stats(self, pred_bboxes, pred_scores, pred_classes, gt_bboxes, gt_classes):
        ious = bbox_utils.bbox_iou(pred_bboxes[:,None], gt_bboxes[None], xywh=False)
        max_iou = tf.reduce_max(ious, -1)
        max_iou_idx = tf.argmax(ious, -1, output_type=tf.int32)
        sorted_idx = tf.argsort(max_iou, direction="DESCENDING")
        u_classes, u_idx, u_count = tf.unique_with_counts(tf.reshape(gt_classes, (-1)))
        for i, u_class, in enumerate(u_classes):
            self.stats[int(u_class)]["total"] += u_count[i]
        
        past_ids = []
        for i, idx in enumerate(sorted_idx):
            if pred_bboxes[i][2] <= pred_bboxes[i][0] or pred_bboxes[i][3] <= pred_bboxes[i][1]:
                continue
            pred_class = int(pred_classes[idx])
            iou = max_iou[idx]
            pred_id = max_iou_idx[idx]
            score = pred_scores[idx]
            
            gt_class = int(gt_classes[idx])
            
            self.stats[pred_class]['scores'].append(score)
            
            if iou > self.iou_threshold and pred_class == gt_class and pred_id not in past_ids:
                self.stats[pred_class]['tp'].append(1)
                self.stats[pred_class]['fp'].append(0)
                past_ids.append(pred_id)
            else:
                self.stats[pred_class]['tp'].append(0)
                self.stats[pred_class]['fp'].append(1)
    
    def calculate_mAP(self):
        aps = []
        for label in self.stats.keys():
            label_stats = self.stats[label]
            ids = np.argsort(-np.array(label_stats['scores']))
            total = label_stats['total']
            
            cumsum_tp = np.cumsum(np.array(label_stats['tp'])[ids])
            cumsum_fp = np.cumsum(np.array(label_stats['fp'])[ids])
            
            recall = cumsum_tp / total
            precision = cumsum_tp / (cumsum_tp + cumsum_fp)
            ap = self.calculate_AP(recall, precision)
        
            self.stats[label]['recall'] = recall
            self.stats[label]['precision'] = precision
            self.stats[label]['ap'] = ap
            aps.append(ap)
        self.mAP = np.mean(aps)
        
    def calculate_AP(self, recall, precision):
        ap = 0 
        for r in np.arange(0, 1.1, 0.1):
            prec_rec = precision[recall >= r]
            if len(prec_rec) > 0:
                ap += np.max(prec_rec)
        ap /= 11
        return ap
    
    def get_result(self):
        text = ''
        class_max_length = np.max(list(map(lambda x: len(x), LABELS)))
        block_max_length = 51 - class_max_length
        for label in self.stats.keys():
            ap = self.stats[label]['ap']
            class_name = self.stats[label]['label']
            block = '■' * int(ap * block_max_length)
            text += f'{class_name:>{class_max_length}}|'
            text += f'{block:<{block_max_length}}|{ap:.2f}\n'
        text += f'mAP{int(self.iou_threshold*100)}: {self.mAP:.4f}'
        return text
        