import shutil, sys, os, tqdm
import numpy as np
import tensorflow as tf
from config import *
from utils import data_utils, train_utils, io_utils
from utils.preset import preset


def main():
    dataloader = data_utils.DataLoader()
    train_dataset = dataloader('train')
    train_dataset_length = dataloader.length('train')//BATCH_SIZE
    valid_dataset = dataloader('valid')
    valid_dataset_length = dataloader.length('valid')//BATCH_SIZE
    
    model, start_epoch, max_loss = train_utils.get_model()
    train_max_loss = valid_max_loss = max_loss
    
    global_step = (start_epoch) * train_dataset_length
    warmup_step = 0
    max_step = EPOCHS * train_dataset_length
    optimizer = tf.keras.optimizers.Adam(decay=0.005)

    train_writer = tf.summary.create_file_writer(LOGDIR)
    val_writer = tf.summary.create_file_writer(LOGDIR)

    for epoch in range(start_epoch, EPOCHS):
        #train
        train_iter, train_loc_loss, train_conf_loss, train_prob_loss, train_total_loss = 0, 0., 0., 0., 0.
        
        train_tqdm = tqdm.tqdm(train_dataset, total=train_dataset_length, desc=f'train epoch {epoch}/{EPOCHS}')
        for batch_data in train_tqdm:
            global_step += 1
            train_iter += 1
            warmup_step += 1
            
            batch_images = batch_data[0]
            batch_labels = batch_data[1:]
            optimizer.lr.assign(train_utils.step_lr_scheduler(global_step, max_step, train_dataset_length, warmup_step))
            
            with tf.GradientTape() as train_tape:
                preds = model(batch_images, True)
                train_loss = model.loss(batch_labels, preds)
                gradients = train_tape.gradient(train_loss[3], model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
            train_loc_loss += train_loss[0]
            train_conf_loss += train_loss[1]
            train_prob_loss += train_loss[2]
            train_total_loss += train_loss[3]
            
            train_loss_ = [train_loc_loss/train_iter, train_conf_loss/train_iter,
                           train_prob_loss/train_iter, train_total_loss/train_iter]
            
            io_utils.write_summary(train_writer, global_step, train_loss_, optimizer.lr.numpy())
            tqdm_text = 'lr={:.5f}, total_loss={:.5f}, loc_loss={:.5f}, conf_loss={:.5f}, prob_loss={:.5f}'\
                        .format(optimizer.lr.numpy(),
                                train_loss_[3].numpy(), train_loss_[0].numpy(), 
                                train_loss_[1].numpy(), train_loss_[2].numpy())
            train_tqdm.set_postfix_str(tqdm_text)
        if train_loss_[3] < train_max_loss:
            train_max_loss = train_loss_[3]
            train_utils.save_model(model, epoch, train_loss_, False)
            
        # valid
        # if epoch % 5 == 0:
        valid_iter, valid_loc_loss, valid_conf_loss, valid_prob_loss, valid_total_loss = 0, 0, 0, 0, 0
        valid_tqdm = tqdm.tqdm(valid_dataset, total=valid_dataset_length, desc=f'valid epoch {epoch}/{EPOCHS}')
        for batch_data in valid_tqdm:
            batch_images = batch_data[0]
            batch_labels = batch_data[1:]
            
            # with tf.GradientTape() as valid_tape:
            preds = model(batch_images)
            valid_loss = model.loss(batch_labels, preds)
                
            valid_loc_loss += valid_loss[0]
            valid_conf_loss += valid_loss[1]
            valid_prob_loss += valid_loss[2]
            valid_total_loss += valid_loss[3]

            valid_iter += 1
            
            
            valid_loss_ = [valid_loc_loss / valid_iter, valid_conf_loss / valid_iter, 
                            valid_prob_loss / valid_iter, valid_total_loss / valid_iter]
            
            tqdm_text = '#total_loss={:.5f}, #loc_loss={:.5f}, #conf_loss={:.5f}, #prob_loss={:.5f}'\
                        .format(valid_loss_[3].numpy(), valid_loss_[0].numpy(), 
                                valid_loss_[1].numpy(), valid_loss_[2].numpy())
            valid_tqdm.set_postfix_str(tqdm_text)
        
        
        io_utils.write_summary(val_writer, epoch, valid_loss_)

        if valid_loss_[3] < valid_max_loss:
            valid_max_loss = valid_loss_[3]
            train_utils.save_model(model, epoch, valid_loss_)

        

if __name__ == '__main__':
    preset()
    main()