# train config
EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-1
WARMUP = 1000
DTYPE = 'voc'
IMAGE_SIZE = 416
MAX_BBOXES = 100
NMS_TYPE = 'soft_gaussian'
LR_SCHEDULER = 'poly'

IOU_THRESHOLD = 0.5
if NMS_TYPE == 'soft_gaussian':
    SCORE_THRESHOLD = 1e-3
else:
    SCORE_THRESHOLD = 0.3
SIGMA = 0.5
COORD = 5
NOOBJ = 0.5
EPS = 1e-7

# model config
MODEL_TYPE = 'YOLOv3'
CHECKPOINTS_DIR = 'checkpoints/' + DTYPE + '/'
TRAIN_CHECKPOINTS_DIR = CHECKPOINTS_DIR + MODEL_TYPE + '/train/'
VALID_CHECKPOINTS_DIR = CHECKPOINTS_DIR + MODEL_TYPE + '/valid/'
CHECKPOINTS = VALID_CHECKPOINTS_DIR + MODEL_TYPE
LOAD_CHECKPOINTS = False
NUM_ANCHORS = 3

# log config
LOGDIR = 'logs/' + MODEL_TYPE + '_' + DTYPE + '_log'

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
        ANCHORS = [[[24, 31], [51, 74], [75, 154]], [[137, 85], [120, 243], [178, 167]], [[327, 147], [220, 289], [351, 286]]]
    elif MODEL_TYPE == 'YOLOv3_tiny':
        STRIDES = [16, 32]
        ANCHORS = [[[28, 37], [72, 91], [102, 187]], [[249, 127], [180, 262], [330, 278]]]
           
elif DTYPE == 'custom':
    LABELS = ['Nam Joo-hyuk', 'Kim Da-mi', 'Kim Seong-cheol', 'Yoo Jae-suk', 
              'Kim Tae-ri', 'Choi Woo-shik']
    NUM_CLASSES = 6
    if MODEL_TYPE == 'YOLOv3':
        STRIDES = [8, 16, 32]
        ANCHORS = [[[31, 35], [47, 50], [59, 67]], [[70, 77], [83, 90], [96, 106]], [[117, 129], [151, 170], [199, 227]]]
    elif MODEL_TYPE == 'YOLOv3_tiny':
        STRIDES = [16, 32]
        ANCHORS = [[[29, 37], [71, 89], [100, 183]], [[244, 125], [179, 259], [331, 273]]]
        
# draw
DRAW = True