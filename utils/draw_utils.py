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

def save_image(image):
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
    
def inference_video(video_path=0):
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    filename = OUTPUT_DIR + 'video/' + 'inference_'
    filename += str(len(glob.glob(filename+'*.mp4')))
    writer = cv2.VideoWriter(filename + '.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    total_frames = 0 
    start_time = time.time()
    prev_time = 0
    while(cv2.waitKey(1) != 27):
        cur_time = time.time()
        
        ret, frame = cap.read()
        if ret:
            total_frames += 1
            # add inference
            writer.write(frame)
            
            sec = cur_time - prev_time
            fps = 1 / sec
            prev_time = cur_time
            text = f'Inference Time: {sec:.3f}, FPS: {fps:.1f}, {frame.shape[1]}, {frame.shape[0]}'
            
            cv2.putText(frame, text, (10,10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,255,255), 1)
            cv2.imshow('Inference', frame)
    end_time = time.time()
    sec = end_time - start_time
    avg_fps = total_frames / sec
    print(f'Average Inferrence Time:{1/avg_fps:.3f}, Average FPS:{avg_fps:.3f}')
    writer.release()
    cap.release()
    cv2.destroyAllWindows()