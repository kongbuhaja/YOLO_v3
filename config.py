# train config
EPOCHS = 500
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
        ANCHORS = [[[23, 32], [54, 68], [68, 147]], [[137, 97], [116, 223], [308, 129]], [[200, 206], [231, 331], [357, 259]]]

elif DTYPE == 'coco':
    pass
