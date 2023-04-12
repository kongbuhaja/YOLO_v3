from config import *
from utils import anchor_utils, io_utils
import numpy as np
import sys, os, cv2, gdown, zipfile, shutil
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

def read_tfrecord(filepath):
    dataset =  tf.data.TFRecordDataset(filepath, num_parallel_reads=-1) \
                    .map(parse_tfrecord_fn)
    return dataset

def generator(data):
    for image_file, labels in data:
        image = read_image(image_file)
        height, width = image.shape[:2]
        yield image, labels, float(width), float(height)
    
def make_tfrecord(data, filepath, infopath):    
    with open(infopath, 'w') as f:
        f.write(str(len(data)))

    print(f'Start make {filepath}......      ', end='', flush=True)
    with tf.io.TFRecordWriter(filepath) as writer:
        for image_file, labels in tqdm.tqdm(data):
            image = read_image(image_file)
            height, width = image.shape[:2]
            writer.write(_data_features(image, labels, float(width), float(height)))
    print('Done!')

def download_dataset(dtype):
    if dtype in ['voc', 'coco']:
        out_dir = './data/' + dtype
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            if dtype == 'voc':
                year = '/2012'
            elif dtype == 'coco':
                year = './2017'
            tfds.load(dtype + year, data_dir=out_dir)
            if os.path.exists(out_dir + '/' + dtype):
                shutil.rmtree(out_dir + '/' + dtype)
            for file in os.listdir(out_dir + '/downloads/'):
                if file.endswith('.tar') or file.endswith('.INFO'):
                    os.remove(out_dir + '/downloads/' + file)
    
    elif dtype == 'custom':
        path = 'https://drive.google.com/uc?id='
        file = '15G2fgzBd8uXPr8yLgcJFhOYfpZlzd294'
        out_dir = './data/' + dtype + '/'
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
            gdown.download(path + file, out_dir + dtype + '.zip')
        
            with zipfile.ZipFile(out_dir + dtype + '.zip', 'r') as zip:
                zip.extractall(out_dir)
            
            os.remove(out_dir + dtype + '.zip')

def make_new_anchors(normalized_wh):
    print(f'Start calulate anchors......      ', end='', flush=True)
    anchors = np.array(ANCHORS)
    new_anchors = anchor_utils.generate_anchors(normalized_wh, np.prod(anchors.shape[:-1]), IMAGE_SIZE, IMAGE_SIZE)
    io_utils.edit_config(str(ANCHORS), str(new_anchors.reshape(anchors.shape).tolist()))
    print('Done!')
    
def read_image(image_file):
    image = cv2.imread(image_file)
    return image[..., ::-1]

def parse_tfrecord_fn(example):
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
    raise Exception(f'Wrong array dtype: {value.dtype}')
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