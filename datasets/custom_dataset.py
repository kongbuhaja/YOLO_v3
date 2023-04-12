import tensorflow as tf
import numpy as np
import os, shutil, sys, cv2
import xml.etree.ElementTree as ET
import tqdm
from config import *
from PIL import Image
from utils import anchor_utils, io_utils
from datasets import common

class Dataset():
    def __init__(self, split, dtype=DTYPE, anchors=ANCHORS, labels=LABELS, image_size=IMAGE_SIZE,
                 create_anchors=CREATE_ANCHORS):
        self.split = split
        self.dtype = dtype
        self.anchors = np.array(anchors)
        self.labels = labels
        self.image_size = image_size
        self.create_anchors = create_anchors
        print(f'Dataset: {self.dtype}')

    def load(self, use_tfrecord=True):
        assert self.split in ['train', 'val', 'test'], 'Check your dataset type and split.'
        common.download_dataset(self.dtype)
        if self.create_anchors:
            data, normalized_wh = self.read_files()
            common.make_new_anchors(normalized_wh)
            print('Anchors are changed. You need to restart file!')
            print('Please restart train.py')
            sys.exit()
        if use_tfrecord:
            filepath = f'./data/{self.dtype}/{self.split}.tfrecord'
            infopath = f'./data/{self.dtype}/{self.split}.txt'
            if os.path.exists(filepath):
                print(f'{filepath} is exist')
            else:
                data, normalized_wh = self.read_files()
                common.make_tfrecord(data, filepath, infopath)                
            return common.read_tfrecord(filepath)
        else:
            data, normalized_wh = self.read_files()
            return tf.data.Dataset.from_generator(common.generator, 
                                                  output_types=(tf.uint8, tf.float32, tf.float32, tf.float32),
                                                  output_shapes=((None, None, 3), (None, 5), (), ()))(data)

    def read_files(self):
        data = []
        normalized_wh = np.zeros((0,2))
        print('Reading local_files...  ', end='', flush=True)
        
        anno_dir, image_dir = self.load_directory(self.split)
        
        anno_files = os.listdir(anno_dir)

        for anno_file in anno_files:
            image_file, labels, labels_wh = self.parse_annotation(anno_dir + anno_file)
            data += [[image_dir + image_file, labels]]
            if self.create_anchors:
                normalized_wh = np.concatenate([normalized_wh, labels_wh], 0)
            
        np.random.shuffle(data)
        print('Done!')
        
        return data, normalized_wh
        
    def load_directory(self, split):
        extracted_dir = './data/' + self.dtype + '/'
        for dir in os.listdir(extracted_dir):
            if split == 'train' and 'tra' in dir:
                    break
            elif split == 'val' and 'val' in dir:
                    break
        anno_dir = extracted_dir + dir + '/Annotations/'
        image_dir = extracted_dir + dir + '/JPEGImages/'
        return anno_dir, image_dir

    def parse_annotation(self, anno_path):
        tree = ET.parse(anno_path)
        labels = []
        labels_wh = np.zeros((0,2))
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                filename = elem.text
            elif 'width' in elem.tag:
                width = float(elem.text)
            elif 'height' in elem.tag:
                height = float(elem.text)
            elif 'object' in elem.tag:
                for attr in list(elem):
                    if 'name' in attr.tag:
                        label = float(self.labels.index(attr.text))
                    elif 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                xmin = float(dim.text)
                            elif 'ymin' in dim.tag:
                                ymin = float(dim.text)
                            elif 'xmax' in dim.tag:
                                xmax = float(dim.text)
                            elif 'ymax' in dim.tag:
                                ymax = float(dim.text)
                        labels.append([xmin, ymin, xmax, ymax, label])
        if self.create_anchors:
            labels_ = np.array(labels)[:,:4]
            length = np.maximum(width, height)
            labels_w = (labels_[:,2:3] - labels_[:,0:1])/length
            labels_h = (labels_[:,3:4] - labels_[:,1:2])/length
            labels_wh = np.concatenate([labels_w, labels_h], -1)
            
        return filename, labels, labels_wh
