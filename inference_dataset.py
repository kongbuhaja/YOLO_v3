from utils import data_utils, train_utils, anchor_utils, post_processing, draw_utils, bbox_utils, eval_utils, io_utils
from utils.preset import preset
from config import *
import tensorflow as tf
import tqdm
import numpy as np


def main():
    anchors = list(map(lambda x: tf.reshape(x,[-1,4]), anchor_utils.get_anchors_xywh(ANCHORS, STRIDES, IMAGE_SIZE)))
    dataloader = data_utils.DataLoader()
    test_dataset = dataloader('valid', use_label='test')
    test_dataset_legnth = dataloader.length('valid')//BATCH_SIZE
    
    model, _, _ = train_utils.get_model()
    
    stats = eval_utils.stats()
    
    all_images = []
    all_grids = []
    all_labels = []
    
    test_tqdm = tqdm.tqdm(test_dataset, total=test_dataset_legnth, desc=f'test data prediction')
    for batch_data in test_tqdm:
        batch_images = batch_data[0]
        batch_labels = batch_data[1]
        
        all_images.append(batch_images.numpy()[...,::-1]*255.)
        all_grids.append(model(batch_images))
        all_labels.append(batch_labels)
        
    # NMS구현해서 넣기

    inference_tqdm = tqdm.tqdm(range(len(all_images)), desc=f'drawing and calculate')
    for i in inference_tqdm:
        batch_images = all_images[i]
        batch_grids = all_grids[i]
        batch_labels = all_labels[i]
        batch_bboxes, batch_scores, batch_classes = post_processing.prediction_to_bbox(batch_grids, anchors)
        for image, bboxes, scores, classes, labels in zip(batch_images.astype(np.uint8), batch_bboxes, batch_scores, batch_classes, batch_labels):
            NMS_bboxes, NMS_scores, NMS_classes = post_processing.NMS(bboxes, scores, classes)
            gt_bboxes = bbox_utils.xywh_to_xyxy(labels[..., :4])
            gt_scores = tf.ones_like(labels[..., 4])
            gt_classes = labels[..., 4]
            if DRAW:
                pred = draw_utils.draw_labels(image.copy(), NMS_bboxes, NMS_scores, NMS_classes)
                origin = draw_utils.draw_labels(image.copy(), gt_bboxes, gt_scores, gt_classes)
                output = np.concatenate([origin, pred], 1)
                draw_utils.show_and_save_image(output)

            stats.update_stats(NMS_bboxes, NMS_scores, NMS_classes, gt_bboxes, gt_classes)
    stats.calculate_mAP()
    evaluation = stats.get_result()
    print(evaluation)
    io_utils.write_eval(evaluation)   
    
    
if __name__ == '__main__':
    preset()
    main()