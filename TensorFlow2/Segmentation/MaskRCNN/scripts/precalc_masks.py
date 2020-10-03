#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append("../")
from mask_rcnn import dataloader_utils as dlu
from mask_rcnn.ops import preprocess_ops
from mask_rcnn.object_detection import preprocessor
from mask_rcnn.object_detection import tf_example_decoder
from absl import flags
import functools
from absl import app
from mask_rcnn.utils.logging_formatter import logging
from tqdm import tqdm

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
OUTPUT_BASE_DIR = os.path.join(BASE_DATA_PATH, "precalc_masks_test")

flags.DEFINE_string('input_file_pattern',
                    default=input_file_pattern,
                    help='input file pattern')

flags.DEFINE_integer('num_files',
                     default=1024,
                     help=('Total number of files to generate.'))

flags.DEFINE_bool('scatter_files',
                  default=False,
                  help='scatter input to num_files files')

flags.DEFINE_string('output_directory',
                    default=OUTPUT_BASE_DIR,
                    help='Output directory to save the files')


def get_new_example(encoded, source_id, height, width, xmin, xmax, ymin, ymax,
                    label, area, is_crowd, scaled_masks, masks,num_scaled_masks):
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
          int64_feature(num_scaled_masks),
      'image/object/mask':
          bytes_list_feature(masks.numpy())
  }
  return tf.train.Example(features=tf.train.Features(
      feature=new_keys_to_features))


#images_to_process = np.load("fast_images.npz")["less_20ms"]
#to_select = set([str(x) for x in images_to_process])
#num_entries = len(to_select)

#entries_per_file = int(num_entries / num_files)
#remainder = num_entries - num_files * entries_per_file
#entries_per_file += 1
# print(
#     f"num_entries={num_entries}, entries_per_file={entries_per_file} max_file={num_entries//entries_per_file}"
# )

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


def _parse_example(example_proto):
  return example_proto, tf.io.parse_single_example(example_proto,
                                                   keys_to_features)


def embed_masks(dataset, output_dir=OUTPUT_BASE_DIR, num_files=1024):
  use_instance_mask = True
  regenerate_source_id = False
  ds_params = {
      "anchor_scale": 8.0,
      "aspect_ratios": [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]],
      "batch_size": 1,
      "gt_mask_size": 112,
      "image_size": [1024, 1024],
      "include_groundtruth_in_features": False,
      "augment_input_data": False,
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
      "preprocessed_data": False,
  }
  ds_params["image_size"] = [832, 1344]

  # parser = functools.partial(dataset_parser,
  #                            mode=tf.estimator.ModeKeys.TRAIN,
  #                            params=ds_params,
  #                            use_instance_mask=True,
  #                            seed=1517)
  decoder = tf_example_decoder.TfExampleDecoder(
      use_instance_mask=use_instance_mask,
      regenerate_source_id=regenerate_source_id,
      append_original=True)
  parser = lambda x: decoder.decode(x)
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  writers = [
      tf.io.TFRecordWriter(
          os.path.join(output_dir,
                       "train-{:04d}-{:04d}.tfrecord".format(i + 1, num_files)))
      for i in range(num_files)
  ]
  proc_counter = 0

  def get_both_flips(boxes, masks):
    fboxes = preprocessor._flip_boxes_left_right(boxes)
    fmasks = preprocessor._flip_masks_left_right(masks)
    return (boxes, fboxes), (masks, fmasks)

  def get_example_from_parsed_tensors(parsed_tensors, scaled_masks):
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
    new_ex = get_new_example(encoded, source_id, height, width, xmin, xmax,
                             ymin, ymax, label, area, is_crowd, scaled_masks,
                             masks,num_scaled_masks=scaled_masks.shape[0]//2)
    return new_ex
  def calculate_masks(x):
    data=decoder.decode(x)
    image = tf.image.convert_image_dtype(data['image'], dtype=tf.float32)
    boxes, classes, indices, instance_masks = dlu.process_boxes_classes_indices_for_training(
        data,
        skip_crowd_during_training=ds_params['skip_crowd_during_training'],
        use_category=ds_params['use_category'],
        use_instance_mask=use_instance_mask)
    b, m = get_both_flips(boxes, instance_masks)
    prep_image, prep_im_info, oscaled_boxes, oscaled_masks = preprocess_ops.resize_and_pad(
        image,
        boxes=b[0],
        masks=m[0],
        target_size=ds_params["image_size"],
        stride=2**ds_params["max_level"])
    _, _, fscaled_boxes, fscaled_masks = preprocess_ops.resize_and_pad(
        image,
        boxes=b[1],
        masks=m[1],
        target_size=ds_params["image_size"],
        stride=2**ds_params["max_level"])
    padded_image_size = prep_image.get_shape().as_list()[:2]
    omask = preprocess_ops.crop_gt_masks(
        instance_masks=oscaled_masks,
        boxes=oscaled_boxes,
        gt_mask_size=ds_params["gt_mask_size"],
        image_size=padded_image_size)
    fmask=preprocess_ops.crop_gt_masks(
        instance_masks=fscaled_masks,
        boxes=fscaled_boxes,
        gt_mask_size=ds_params["gt_mask_size"],
        image_size=padded_image_size)
    scaled_masks=tf.concat([omask,fmask],axis=0)
    parsed_tensors = data["parsed_tensors"]
    return parsed_tensors,scaled_masks
  parsed_dataset = dataset.map(
      calculate_masks, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(1000)



  p_bar = tqdm(range(118288), file=sys.stdout, 
                         bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
  pit=iter(p_bar)
  for parsed_tensors,scaled_masks in parsed_dataset:
    new_ex = get_example_from_parsed_tensors(parsed_tensors, scaled_masks)
    writers[proc_counter % num_files].write(new_ex.SerializeToString())
    proc_counter += 1
    _=next(pit)
    p_bar.set_description("image: {0:7d}".format(proc_counter))
    # if proc_counter % 1000 == 0:
    #   print("processed", proc_counter, "images", "scaled_masks.shape=",
    #         scaled_masks.shape)
    # if proc_counter == 100:
    #   break

  for w in writers:
    w.close()


def scatter_dataset(dataset, output_dir, num_new_files):
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  writers = [
      tf.io.TFRecordWriter(
          os.path.join(
              output_dir,
              "train-{:04d}-{:04d}.tfrecord".format(i + 1, num_new_files)))
      for i in range(num_new_files)
  ]
  for i, ex in enumerate(dataset):
    writers[i % num_new_files].write(ex.numpy())
    if ((i + 1) % 1000) == 0:
      logging.info(f"processed {i} images")
  logging.info(f"processed {i} images")
  for w in writers:
    w.close()


def main(argv):
  del argv
  FLAGS = flags.FLAGS
  dataset = tf.data.Dataset.list_files(FLAGS.input_file_pattern, shuffle=False)

  record_reader = tf.data.TFRecordDataset(dataset,
                                          num_parallel_reads=tf.data.AUTOTUNE,
                                          buffer_size=5000000000)
  if FLAGS.scatter_files:
    scatter_dataset(record_reader, FLAGS.output_directory, FLAGS.num_files)
  else:
    embed_masks(record_reader, FLAGS.output_directory, FLAGS.num_files)


if __name__ == '__main__':
  app.run(main)
