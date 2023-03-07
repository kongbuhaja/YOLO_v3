from utils import data_utils, train_utils, anchor_utils, post_processing, draw_utils, bbox_utils
from utils.preset import preset
from config import *
import tensorflow as tf
import tqdm
import numpy as np
import time

def main():
    anchors = list(map(lambda x: tf.reshape(x,[-1,4]), anchor_utils.get_anchors_xywh(ANCHORS, STRIDES, IMAGE_SIZE)))
    dataloader = data_utils.DataLoader()
    test_dataset = dataloader('valid', use_label='test')
    test_dataset_legnth = dataloader.length('valid')//BATCH_SIZE
    
    model, _, _ = train_utils.get_model()
    
    all_images = []
    all_grids = []
    all_labels = []
    
    all_bboxes = []
    all_scores = []
    all_classes = []
    
    test_tqdm = tqdm.tqdm(test_dataset, total=test_dataset_legnth, desc=f'test data prediction')
    for batch_data in test_tqdm:
        batch_images = batch_data[0]
        batch_labels = batch_data[1]
        
        all_images.append(batch_images.numpy()*255.)
        all_grids.append(model(batch_images))
        all_labels.append(batch_labels)
        
    # NMS구현해서 넣기
    for batch_images, batch_grids, batch_labels in zip(all_images, all_grids, all_labels):
        batch_bboxes, batch_scores, batch_classes = post_processing.prediction_to_bbox(batch_grids, anchors)
        for image, bboxes, scores, classes, labels in zip(batch_images.astype(np.uint8), batch_bboxes, batch_scores, batch_classes, batch_labels):
            NMS_bboxes, NMS_scores, NMS_classes = post_processing.NMS(bboxes, scores, classes)
            all_bboxes.append(NMS_bboxes)
            all_scores.append(NMS_scores)
            all_classes.append(NMS_classes)
            pred = draw_utils.draw_labels(image.copy(), NMS_bboxes, NMS_scores, NMS_classes)
            origin = draw_utils.draw_labels(image.copy(), bbox_utils.xywh_to_xyxy(labels[..., :4]), tf.ones_like(labels[..., 4]), labels[..., 4])
            output = np.concatenate([origin, pred], 1)
            draw_utils.inference_image(output.astype(np.uint8))
    
    
    
    
if __name__ == '__main__':
    preset()
    main()