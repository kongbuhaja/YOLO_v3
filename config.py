# train config
EPOCHS = 600
BATCH_SIZE = 16
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
CAM_FPS = 30
CAM_WIDTH = 1920
CAM_HEIGHT = 1080

if DTYPE =='voc':
    LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person',
              'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    NUM_CLASSES = 20
    if MODEL_TYPE == 'YOLOv3':
        STRIDES = [8, 16, 32]
        NUM_ANCHORS = 3
        ANCHORS = [[[25, 30], [49, 74], [120, 85]], [[73, 157], [131, 217], [247, 124]], [[209, 278], [333, 199], [341, 312]]]
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
        ANCHORS = [[[24, 27], [35, 41], [45, 53]], [[56, 64], [67, 77], [79, 93]], [[97, 112], [121, 146], [177, 200]]]
    elif MODEL_TYPE == 'YOLOv3_tiny':
        STRIDES = [16, 32]
        NUM_ANCHORS = 2
        YOLO_ANCHORS = [[[11, 14], [23, 27], [37, 58]], [[81, 82], [135, 169], [344, 319]]]