import cv2, os, glob, sys, time
import numpy as np
from config import *

def draw_labels(image, bboxes, scores, classes):
    for bbox, score, cls in zip(bboxes.numpy().astype(np.int32), scores.numpy(), classes.numpy().astype(np.int32)):
        if np.sum(bbox)==0:
            break;
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.rectangle(image, bbox[:2], bbox[2:], color, 2)
        cv2.putText(image, f'{LABELS[cls]}:{score:.3f}', (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, color, 1)
    return image

def show_and_save_image(image):
    if image.shape[1] != IMAGE_SIZE:
        title = 'truth_and_pred'
    else:
        title = 'prediction'
    
    filename = OUTPUT_DIR + 'image/' + title
    filename += '_' + str(len(glob.glob(filename + '*.jpg')))
    
    cv2.imshow(title, image)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
        sys.exit()
    elif key == ord('s'):
        cv2.imwrite(filename + '.jpg', image)
    cv2.destroyWindow(title)
    