import tensorflow as tf
from config import *
from utils import io_utils

def lr_scheduler(epoch, warmup_iter):
    if epoch < 100:
        lr = LR
    elif epoch < 180:
        lr = LR
    elif epoch < 200:
        lr = LR*0.5
    elif epoch < 250:
        lr = LR*0.01
    else:
        lr = LR*0.01
    if warmup_iter < WARMUP:
        lr = lr / WARMUP * (warmup_iter+1)
    return lr
            
def load_model(model):
    model.load_weights(CHECKPOINTS)
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