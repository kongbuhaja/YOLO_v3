import tensorflow as tf
import numpy as np
import os, shutil, sys, cv2
import xml.etree.ElementTree as ET
import tqdm
from config import *
from PIL import Image
from utils import anchor_utils, io_utils

class Dataset():
    def __init__(self, split, dtype=DTYPE, anchors=ANCHORS, labels=LABELS, image_size=IMAGE_SIZE):
        self.split = split
        self.dtype = dtype
        self.anchors = np.array(anchors)
        self.labels = labels
        self.image_size = image_size
        self.data = []
        self.normalized_anno = np.zeros((0,2))
        self.new_anchors = False

    def load(self):
        assert self.split in ['train', 'valid'], "Check your dataset type and split."
        self.read_files()

    def generator(self):
        for image_file, labels in self.data:
            image = self.read_image(image_file)
            yield image, labels

    def read_image(self, image_file):
        image = cv2.imread(image_file)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def read_files(self):
        directories = self.get_load_directory(self.split)
        print('Reading local_files...  ', end='', flush=True)
        for directory in directories:
            image_dir = directory + '/JPEGImages/'
            anno_dir = directory + '/Annotations/'

            image_files = os.listdir(image_dir)
            anno_files = os.listdir(anno_dir) if self.split != 'test' else []
            for i in range(len(image_files)):
                image_file = image_dir + image_files[i]

                if self.split == 'test':
                    labels = []
                else:
                    anno_file = anno_dir + anno_files[i]
                    labels, width, height = self.parse_annotation(anno_file)
                self.data.append([image_file, labels, width, height])
        np.random.shuffle(self.data)
        print('Done!')

    def get_load_directory(self, split):
        load_directory=[]
        extracted_dir = "./data/" + self.dtype + "/"
        for dir in os.listdir(extracted_dir):
            if split=="train":
                if "tra" in dir:
                    load_directory.append(extracted_dir + dir)
            elif split=="valid":
                if "val" in dir:
                    load_directory.append(extracted_dir + dir)
        return load_directory

    def parse_annotation(self, anno_path):
        tree = ET.parse(anno_path)
        labels = []
        
        for elem in tree.iter():
            if "width" in elem.tag:
                width = float(elem.text)
            elif "height" in elem.tag:
                height = float(elem.text)
            elif "object" in elem.tag:
                for attr in list(elem):
                    if "name" in attr.tag:
                        label = float(self.labels.index(attr.text))
                    elif "bndbox" in attr.tag:
                        for dim in list(attr):
                            if "xmin" in dim.tag:
                                xmin = float(dim.text)
                            elif "ymin" in dim.tag:
                                ymin = float(dim.text)
                            elif "xmax" in dim.tag:
                                xmax = float(dim.text)
                            elif "ymax" in dim.tag:
                                ymax = float(dim.text)
                        labels.append([xmin, ymin, xmax, ymax, label])
        if self.new_anchors:
            labels_ = np.array(labels)[:,:4]
            length = np.maximum(width, height)
            labels_w = (labels_[:,2:3] - labels_[:,0:1])/length
            labels_h = (labels_[:,3:4] - labels_[:,1:2])/length
            labels_wh = np.concatenate([labels_w, labels_h], -1)
            self.normalized_anno = np.concatenate([self.normalized_anno, labels_wh], 0)
        return labels, width, height
    def make_new_anchors(self):
        if self.new_anchors:
            print(self.normalized_anno.shape)
            anchors = anchor_utils.generate_anchors(self.normalized_anno, np.prod(self.anchors.shape[:-1]), self.image_size, self.image_size)
            io_utils.edit_config(str(self.anchors.tolist()), str(anchors.reshape(self.anchors.shape).tolist()))
        
    def make_tfrecord(self):
        filepath = f'./data/{self.dtype}/{self.split}.tfrecord'
        infopath = f'./data/{self.dtype}/{self.split}.txt'
        
        if os.path.exists(filepath):
            print(f'{filepath} is exist')
            return 
        
        if self.split == 'train':
            self.new_anchors = True
        self.load()
        self.make_new_anchors()
        
        with open(infopath, 'w') as f:
            f.write(str(len(self.data)))
            

        print(f'Start make {filepath}......      ', end='', flush=True)
        with tf.io.TFRecordWriter(filepath) as writer:
            for image_file, labels, width, height in tqdm.tqdm(self.data):
                image = self.read_image(image_file)
                writer.write(_data_features(image, labels, width, height))
        print('Done!')
        if self.new_anchors:
            print('Anchors are changed. You need to restart file!')
            print("If you don't make new tfrecord, Anchors are't changed")
            sys.exit()

    def read_tfrecord(self):
        filepath = f'./data/{self.dtype}/{self.split}.tfrecord'
        dataset =  tf.data.TFRecordDataset(filepath, num_parallel_reads=-1) \
                        .map(self.parse_tfrecord_fn)
        return dataset

    def parse_tfrecord_fn(self, example):
        feature_description={
            'image': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.VarLenFeature(tf.float32),
            'width': tf.io.FixedLenFeature([], tf.float32),
            'height': tf.io.FixedLenFeature([], tf.float32)
        }
        example = tf.io.parse_single_example(example, feature_description)
        example['image'] = tf.io.decode_jpeg(example['image'], channels=3)
        example['labels'] = tf.reshape(tf.sparse.to_dense(example['labels']), (-1, 5))

        return example['image'], example['labels'], example['width'], example['height']


def _image_feature(value):
    return _bytes_feature(tf.io.encode_jpeg(value).numpy())
def _array_feature(value):
    if 'float' in value.dtype.name:
        return _float_feature(np.reshape(value, (-1)))
    elif 'int' in value.dtype.name:
        return _int64_feature(np.reshape(value, (-1)))
    raise Exception(f"Wrong array dtype: {value.dtype}")
def _string_feature(value):
    return _bytes_feature(value.encode('utf-8'))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    if type(value) == float:
        value=[value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _int64_feature(value):
    if type(value) == int:
        value=[value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _data_features(image, labels, width, height):
    image_feature = _image_feature(image)
    labels_feature = _array_feature(np.array(labels))
    width_feature = _float_feature(width)
    height_feature = _float_feature(height)
    
    objects_features = {
        'image': image_feature,
        'labels': labels_feature,
        'width': width_feature,
        'height': height_feature
    }      
    example=tf.train.Example(features=tf.train.Features(feature=objects_features))
    return example.SerializeToString()