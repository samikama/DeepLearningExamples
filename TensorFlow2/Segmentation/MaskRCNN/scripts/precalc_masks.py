#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append("../")
from mask_rcnn.dataloader_utils import dataset_parser

import functools


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


BASE_DATA_PATH = "/data/coco/coco-2017"
input_file_pattern = f"{BASE_DATA_PATH}/nv_coco/train*.tfrecord"
OUTPUT_BASE_DIR = os.path.join(BASE_DATA_PATH, "precalc_masks")
if not os.path.exists(OUTPUT_BASE_DIR):
  os.mkdir(OUTPUT_BASE_DIR)
dataset = tf.data.Dataset.list_files(input_file_pattern, shuffle=False)
keys_to_features = {
    'image/encoded': tf.io.FixedLenFeature((), tf.string),
    'image/source_id': tf.io.FixedLenFeature((), tf.string),
    'image/height': tf.io.FixedLenFeature((), tf.int64),
    'image/width': tf.io.FixedLenFeature((), tf.int64),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/area': tf.io.VarLenFeature(tf.float32),
    'image/object/is_crowd': tf.io.VarLenFeature(tf.int64),
}


def get_new_example(encoded, source_id, height, width, xmin, xmax, ymin, ymax,
                    label, area, is_crowd, scaled_masks, masks):
  new_keys_to_features = {
      'image/encoded':
          bytes_feature(encoded.numpy()),
      'image/source_id':
          bytes_feature(source_id.numpy()),
      'image/height':
          int64_feature(height.numpy()),
      'image/width':
          int64_feature(width.numpy()),
      'image/object/bbox/xmin':
          float_list_feature(xmin.numpy()),
      'image/object/bbox/xmax':
          float_list_feature(xmax.numpy()),
      'image/object/bbox/ymin':
          float_list_feature(ymin.numpy()),
      'image/object/bbox/ymax':
          float_list_feature(ymax.numpy()),
      'image/object/class/label':
          int64_list_feature(label.numpy()),
      'image/object/area':
          float_list_feature(area.numpy()),
      'image/object/is_crowd':
          int64_list_feature(is_crowd.numpy()),
      'image/scaled_masks':
          float_list_feature(scaled_masks.numpy().reshape([-1])),
      'image/num_scaled_masks':
          int64_feature(scaled_masks.numpy().shape[0]),
      'image/object/mask':
          bytes_list_feature(masks.numpy())
  }
  return tf.train.Example(features=tf.train.Features(
      feature=new_keys_to_features))


use_instance_mask = False
regenerate_source_id = False
ds_params = {
    "anchor_scale": 8.0,
    "aspect_ratios": [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]],
    "batch_size": 1,
    "gt_mask_size": 112,
    "image_size": [1024, 1024],
    "include_groundtruth_in_features": False,
    "augment_input_data": True,
    "max_level": 6,
    "min_level": 2,
    "num_classes": 91,
    "num_scales": 1,
    "rpn_batch_size_per_im": 256,
    "rpn_fg_fraction": 0.5,
    "rpn_min_size": 0.,
    "rpn_nms_threshold": 0.7,
    "rpn_negative_overlap": 0.3,
    "rpn_positive_overlap": 0.7,
    "rpn_post_nms_topn": 1000,
    "rpn_pre_nms_topn": 2000,
    "skip_crowd_during_training": True,
    "use_category": True,
    "visualize_images_summary": False,
    "disable_options": False,
    "preprocessed_data": False
}
ds_params["image_size"] = [832, 1344]

parser = functools.partial(dataset_parser,
                           mode=tf.estimator.ModeKeys.TRAIN,
                           params=ds_params,
                           use_instance_mask=True,
                           seed=1517)
record_reader = tf.data.TFRecordDataset(dataset,
                                        num_parallel_reads=tf.data.AUTOTUNE,
                                        buffer_size=5000000000)
images_to_process = np.load("fast_images.npz")["less_20ms"]
to_select = set([str(x) for x in images_to_process])
num_entries = len(to_select)

num_files = 256
entries_per_file = int(num_entries / num_files)
remainder = num_entries - num_files * entries_per_file
entries_per_file += 1
print(
    f"num_entries={num_entries}, entries_per_file={entries_per_file} max_file={num_entries//entries_per_file}"
)


def _parse_example(example_proto):
  return example_proto, tf.io.parse_single_example(example_proto,
                                                   keys_to_features)


parsed_dataset = record_reader.map(
    parser, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(1000)
writers = [
    tf.io.TFRecordWriter(
        os.path.join(OUTPUT_BASE_DIR,
                     "train-{:04d}-{:04d}.tfrecord".format(i, num_files)))
    for i in range(num_files)
]
proc_counter = 0
for features, labels in parsed_dataset:
  parsed_tensors = features["OrigData"]["parsed_tensors"]
  scaled_masks = labels["cropped_unpadded_gt_masks"]
  encoded = parsed_tensors["image/encoded"]
  source_id = parsed_tensors['image/source_id']
  height = parsed_tensors['image/height']
  width = parsed_tensors['image/width']
  xmin = parsed_tensors['image/object/bbox/xmin']
  xmax = parsed_tensors['image/object/bbox/xmax']
  ymin = parsed_tensors['image/object/bbox/ymin']
  ymax = parsed_tensors['image/object/bbox/ymax']
  label = parsed_tensors['image/object/class/label']
  #print(scaled_masks.shape,scaled_masks.dtype,label.shape,label.shape[0])
  area = parsed_tensors['image/object/area']
  is_crowd = parsed_tensors['image/object/is_crowd']
  masks = parsed_tensors['image/object/mask']
  new_ex = get_new_example(encoded, source_id, height, width, xmin, xmax, ymin,
                           ymax, label, area, is_crowd, scaled_masks, masks)
  writers[proc_counter % num_files].write(new_ex.SerializeToString())
  proc_counter += 1
  if proc_counter % 1000 == 0:
    print("processed", proc_counter, "images", "scaled_masks.shape=",
          scaled_masks.shape)

for w in writers:
  w.close()
