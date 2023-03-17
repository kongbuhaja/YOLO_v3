# train config
EPOCHS = 600
BATCH_SIZE = 8
LR = 1e-1
WARMUP = 10000
DTYPE = 'voc'
IMAGE_SIZE = 416
MAX_BBOXES = 100
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.3
COORD = 5
NOOBJ = 0.5
EPS = 1e-7

# model config
MODEL_TYPE = 'YOLOv3'
CHECKPOINTS_DIR = 'checkpoints/' + DTYPE + '/'
TRAIN_CHECKPOINTS_DIR = CHECKPOINTS_DIR + MODEL_TYPE + '/train/'
VALID_CHECKPOINTS_DIR = CHECKPOINTS_DIR + MODEL_TYPE + '/valid/'
CHECKPOINTS = VALID_CHECKPOINTS_DIR + MODEL_TYPE
LOAD_CHECKPOINTS = True

# log config
LOGDIR = MODEL_TYPE + '_' + DTYPE + '_log'

# inference config
OUTPUT_DIR = 'outputs/'

# cam config
VIDEO_PATH = 0

if DTYPE =='voc':
    LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person',
              'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    NUM_CLASSES = 20
    if MODEL_TYPE == 'YOLOv3':
        STRIDES = [8, 16, 32]
        NUM_ANCHORS = 3
        ANCHORS = [[[24, 30], [46, 72], [123, 75]], [[72, 147], [171, 145], [118, 232]], [[324, 151], [212, 277], [347, 288]]]
    elif MODEL_TYPE == 'YOLOv3_tiny':
        STRIDES = [16, 32]
        NUM_ANCHORS = 2
        YOLO_ANCHORS = [[[10, 14], [23, 27], [37, 58]], [[81, 82], [135, 169], [344, 319]]]
           
elif DTYPE == 'coco':
    pass

elif DTYPE == 'custom':
    LABELS = ['Nam Joo-hyuk', 'Kim Da-mi', 'Kim Seong-cheol', 'Yoo Jae-suk', 
              'Kim Tae-ri', 'Choi Woo-shik']
    NUM_CLASSES = 6
    if MODEL_TYPE == 'YOLOv3':
        STRIDES = [8, 16, 32]
        NUM_ANCHORS = 3
        ANCHORS = [[[25, 29], [38, 46], [50, 58]], [[63, 73], [78, 91], [96, 111]], [[118, 144], [153, 171], [204, 237]]]
    elif MODEL_TYPE == 'YOLOv3_tiny':
        STRIDES = [16, 32]
        NUM_ANCHORS = 2
        YOLO_ANCHORS = [[[11, 14], [23, 27], [37, 58]], [[81, 82], [135, 169], [344, 319]]]
        
        
# draw
DRAW = True