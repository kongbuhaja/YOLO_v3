import os
import tensorflow as tf
from config import *

def write_model_info(checkpoints, epoch, max_loss):
    with open(checkpoints + '.info', 'w') as f:
        text = f'epoch:{epoch}\n'
        text += f'total_loss:{max_loss[3]}\n'
        text += f'loc_loss:{max_loss[0]}\n'
        text += f'conf_loss:{max_loss[1]}\n'
        text += f'prob_loss:{max_loss[2]}\n'
        f.write(text)
        
def read_model_info():
    saved_parameter = {}
    with open(CHECKPOINTS + '.info', 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, value = line[:-1].split(':')
            if key == 'epoch':
                saved_parameter[key] = int(value)
            else:
                saved_parameter[key] = float(value)
    return saved_parameter

def write_summary(writer, step, loss, lr=None):
    with writer.as_default():
        if lr != None:
            tf.summary.scalar('lr', lr, step=step)
            tf.summary.scalar('train_loss/loc_loss', loss[0], step=step)
            tf.summary.scalar('train_loss/conf_loss', loss[1], step=step)
            tf.summary.scalar('train_loss/prob_loss', loss[2], step=step)
            tf.summary.scalar('train_loss/total_loss', loss[3], step=step)
            
        else:
            tf.summary.scalar("validate_loss/loc_loss", loss[0], step=step)
            tf.summary.scalar("validate_loss/conf_loss", loss[1], step=step)
            tf.summary.scalar("validate_loss/prob_loss", loss[2], step=step)
            tf.summary.scalar("validate_loss/total_loss", loss[3], step=step)
        
    writer.flush()
    
def edit_config(pre_text, new_text):
    with open('config.py', 'r') as f:
        lines = f.readlines()
    for l in range(len(lines)):
        if pre_text in lines[l]:
            sp_line = lines[l].split(pre_text)
            lines[l] = sp_line[0] + str(new_text) + sp_line[1]
    with open('config.py', 'w') as f:
        f.writelines(lines)