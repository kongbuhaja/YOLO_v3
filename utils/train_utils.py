import tensorflow as tf
from config import *
from utils import io_utils

def lr_scheduler(step, max_step, step_per_epoch, warmup_step):
    if warmup_step < WARMUP:
        lr = warmup_lr_scheduler(warmup_step, lr=LR)
    elif LR_SCHEDULER == 'step':
        lr = step_lr_scheduler(step, step_per_epoch)
    elif LR_SCHEDULER == 'poly':
        lr = poly_lr_scheduler(step, max_step)
    return lr

def step_lr_scheduler(step, step_per_epoch):    
    epoch = step / step_per_epoch
    
    if epoch < 100:
        lr = LR 
    elif epoch < 300:
        lr = LR * 0.1
    elif epoch < 400:
        lr = LR * 0.5
    elif epoch < 500:
        lr = LR * 0.01
    else:
        lr = LR * 0.05

    return lr

def poly_lr_scheduler(step, max_step, init_lr=LR, power=0.9):
    lr = init_lr*(1-(step/(max_step+1)))**power
    return lr
            
def warmup_lr_scheduler(warmup_step, lr=LR):
    lr = lr / WARMUP * (warmup_step+1)
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