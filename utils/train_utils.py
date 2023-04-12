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
    
    if epoch < 180:
        lr = LR 
    elif epoch < 300:
        lr = LR * 0.5
    elif epoch < 400:
        lr = LR * 0.1
    elif epoch < 500:
        lr = LR * 0.05
    else:
        lr = LR * 0.01

    return lr

def poly_lr_scheduler(step, max_step, init_lr=LR, power=0.9):
    lr = init_lr*(1-(step/(max_step+1)))**power
    return lr
            
def warmup_lr_scheduler(warmup_step, lr=LR):
    lr = lr / WARMUP * (warmup_step+1)
    return lr

def load_model(model):
    model.load_weights(CHECKPOINTS)
    saved = io_utils.read_model_info()
    return model, saved['epoch'], saved['mAP'], saved['total_loss']

def get_model():
    if MODEL_TYPE == 'YOLOv3':
        from models.yolov3 import YOLO
    elif MODEL_TYPE == 'YOLOv3_tiny':
        from models.yolov3_tiny import YOLO
    
    if LOAD_CHECKPOINTS:
        return load_model(YOLO())
    return YOLO(), 0, 0., 1e+50

def save_model(model, epoch, mAP, loss, dir_path):
    checkpoints = dir_path + MODEL_TYPE

    model.save_weights(checkpoints)
    io_utils.write_model_info(checkpoints, epoch, mAP, loss)