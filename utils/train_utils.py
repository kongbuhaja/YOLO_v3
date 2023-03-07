import tensorflow as tf
from config import *
from utils import io_utils

def lr_scheduler(epoch, warmup_iter):
    if warmup_iter < WARMUP:
        return LR / WARMUP * (warmup_iter+1)
    if epoch < 50:
        return LR
    elif epoch < 100:
        return LR
    elif epoch < 150:
        return LR
    elif epoch < 200:
        return LR
    elif epoch < 400:
        return LR
    else:
        return LR*0.1
            
def load_model(model):
    model.load_weights(CHECKPOINTS)
    out = []
    saved_parameter = io_utils.read_model_info()
    return model, saved_parameter['epoch'], saved_parameter['total_loss']

def get_model():
    if MODEL_TYPE == 'YOLOv3':
        from models.yolov3 import Model
    elif MODEL_TYPE == 'YOLOv3_tiny':
        from models.yolov3_tiny import Model
    
    if LOAD_CHECKPOINTS:
        return load_model(Model())
    return Model(), 0, 1e+50

def save_model(model, epoch, max_loss, valid=True):
    if valid:
        checkpoints = VALID_CHECKPOINTS_DIR + MODEL_TYPE
    else:
        checkpoints = TRAIN_CHECKPOINTS_DIR + MODEL_TYPE

    model.save_weights(checkpoints)
    io_utils.write_model_info(checkpoints, epoch, max_loss)