#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data loader and processing.

Defines input_fn of Mask-RCNN for TF Estimator. The input_fn includes training
data for category classification, bounding box regression, and number of
positive examples to normalize the loss during training.

"""
import functools
import math
import multiprocessing
import glob
from mpi4py import MPI
import tensorflow as tf

from mask_rcnn.utils.logging_formatter import logging

from mask_rcnn.utils.distributed_utils import MPI_is_distributed
from mask_rcnn.utils.distributed_utils import MPI_rank_and_size
from mask_rcnn.utils.distributed_utils import MPI_rank
from mask_rcnn.utils.distributed_utils import MPI_size

# common functions
from mask_rcnn.dataloader_utils import dataset_parser

from distutils.version import LooseVersion


class InputReader(object):
  """Input reader for dataset."""

  def __init__(self,
               file_pattern,
               mode=tf.estimator.ModeKeys.TRAIN,
               num_examples=0,
               use_fake_data=False,
               use_instance_mask=False,
               seed=None,
               disable_options=False):

    self._mode = mode
    self._file_pattern = file_pattern
    self._num_examples = num_examples
    self._use_fake_data = use_fake_data
    self._use_instance_mask = use_instance_mask
    self._seed = seed
    self._disable_options = disable_options

  def _create_dataset_parser_fn(self, params):
    """Create parser for parsing input data (dictionary)."""

    return functools.partial(dataset_parser,
                             mode=self._mode,
                             params=params,
                             use_instance_mask=self._use_instance_mask,
                             seed=self._seed)

  def __call__(self, params, input_context=None):

    batch_size = params['batch_size'] if 'batch_size' in params else 1
    do_dist_eval = params.get('dist_eval', False)

    for i in sorted(params.items(), key=lambda x: x[0]):
      logging.info(f"{i[0]}: {i[1]}")

    try:
      seed = params['seed'] if not MPI_is_distributed(
      ) else params['seed'] * MPI_rank()
    except (KeyError, TypeError):
      seed = None

    if MPI_is_distributed():
      n_gpus = MPI_size()

    elif input_context is not None:
      n_gpus = input_context.num_input_pipelines

    else:
      n_gpus = 1

    ##################################################
    #This style of dataset sharding currently fails
    #With more than 32 ranks on evaluation.
    #When MPI_size>32 and running eval, use this
    #simpler pipeline.
    #################################################
    '''if do_dist_eval and n_gpus>32 and \
            (self._mode == tf.estimator.ModeKeys.PREDICT or \
             self._mode == tf.estimator.ModeKeys.EVAL):
            files = glob.glob(self._file_pattern)
            dataset = tf.data.TFRecordDataset(files)
            _shard_idx, _num_shards = MPI_rank_and_size()
            dataset = dataset.shard(_num_shards, _shard_idx)
            parser = lambda x: dataset_parser(x, self._mode, params, self._use_instance_mask, seed=seed)
            dataset = dataset.map(parser , num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(batch_size=batch_size,drop_remainder=False)
            dataset = dataset.repeat()
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return dataset'''

    ##################################################

    dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=False)

    if self._mode == tf.estimator.ModeKeys.TRAIN:

      if input_context is not None:
        logging.info("Using Dataset Sharding with TF Distributed")
        _num_shards = input_context.num_input_pipelines
        _shard_idx = input_context.input_pipeline_id

      elif MPI_is_distributed():
        logging.info("Using Dataset Sharding with Horovod")
        _shard_idx, _num_shards = MPI_rank_and_size()

      try:
        dataset = dataset.shard(num_shards=_num_shards, index=_shard_idx)
        dataset = dataset.shuffle(math.ceil(256 / _num_shards))

      except NameError:  # Not a distributed training setup
        pass
    elif do_dist_eval and (self._mode == tf.estimator.ModeKeys.PREDICT or
                           self._mode == tf.estimator.ModeKeys.EVAL):
      # 32 validation tf records - distribute on upto 32 workers
      if MPI_is_distributed():
        logging.info("Using Evaluation Dataset Sharding with Horovod")
        _shard_idx, _num_shards = MPI_rank_and_size()
        max_shards = min(_num_shards, 32)
        try:
          dataset = dataset.shard(num_shards=max_shards,
                                  index=_shard_idx % max_shards)
        except NameError:  # Not a distributed training setup
          pass

    def _prefetch_dataset(filename):
      return tf.data.TFRecordDataset(filename).prefetch(1)

    dataset = dataset.interleave(
        map_func=_prefetch_dataset,
        cycle_length=32,
        block_length=64,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    if self._num_examples is not None and self._num_examples > 0:
      logging.info("[*] Limiting the amount of sample to: %d" %
                   self._num_examples)
      dataset = dataset.take(self._num_examples)

    dataset = dataset.cache()
    if self._mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(buffer_size=4096,
                                reshuffle_each_iteration=True,
                                seed=seed)

      dataset = dataset.repeat()

    # Parse the fetched records to input tensors for model function.
    dataset = dataset.map(
        map_func=self._create_dataset_parser_fn(params),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    if self._use_fake_data:
      # Turn this dataset into a semi-fake dataset which always loop at the
      # first batch. This reduces variance in performance and is useful in
      # testing.
      logging.info("Using Fake Dataset Loop...")
      dataset = dataset.take(1).cache().repeat()

      if self._mode != tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.take(int(5000 / batch_size))

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE,)
    '''if self._mode == tf.estimator.ModeKeys.PREDICT or n_gpus > 1:
            if not tf.distribute.has_strategy():
                dataset = dataset.apply(
                    tf.data.experimental.prefetch_to_device(
                        '/gpu:0',  # With Horovod the local GPU is always 0
                        buffer_size=1,
                    )
                )'''
    if not self._disable_options:
      data_options = tf.data.Options()

      data_options.experimental_deterministic = seed is not None
      if LooseVersion(tf.__version__) <= LooseVersion("2.0.0"):
        data_options.experimental_distribute.auto_shard = False
      else:
        data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
      # data_options.experimental_distribute.auto_shard = False
      data_options.experimental_slack = params.get('data_slack', False)

      #data_options.experimental_threading.max_intra_op_parallelism = 1
      #data_options.experimental_threading.private_threadpool_size = 5

      # ================= experimental_optimization ================= #

      data_options.experimental_optimization.apply_default_optimizations = False

      # data_options.experimental_optimization.autotune = True
      data_options.experimental_optimization.filter_fusion = True
      data_options.experimental_optimization.map_and_batch_fusion = True
      data_options.experimental_optimization.map_and_filter_fusion = True
      data_options.experimental_optimization.map_fusion = True
      data_options.experimental_optimization.map_parallelization = True

      map_vectorization_options = tf.data.experimental.MapVectorizationOptions()
      map_vectorization_options.enabled = True
      map_vectorization_options.use_choose_fastest = True

      data_options.experimental_optimization.map_vectorization = map_vectorization_options

      data_options.experimental_optimization.noop_elimination = True
      data_options.experimental_optimization.parallel_batch = True
      data_options.experimental_optimization.shuffle_and_repeat_fusion = True

      # ========== Stats on TF Data =============
      # aggregator = tf.data.experimental.StatsAggregator()
      # data_options.experimental_stats.aggregator = aggregator
      # data_options.experimental_stats.latency_all_edges = True

      dataset = dataset.with_options(data_options)

    return dataset


if __name__ == "__main__":
  '''
    Data Loading Benchmark Usage:

    # Real Data - Training
    python -m mask_rcnn.dataloader \
        --data_dir="/data/" \
        --batch_size=2 \
        --warmup_steps=200 \
        --benchmark_steps=2000 \
        --training

    # Real Data - Inference
    python -m mask_rcnn.dataloader \
        --data_dir="/data/" \
        --batch_size=8 \
        --warmup_steps=200 \
        --benchmark_steps=2000

    # --------------- #

    # Synthetic Data - Training
    python -m mask_rcnn.dataloader \
        --data_dir="/data/" \
        --batch_size=2 \
        --warmup_steps=200 \
        --benchmark_steps=2000 \
        --training \
        --use_synthetic_data

    # Synthetic Data - Inference
    python -m mask_rcnn.dataloader \
        --data_dir="/data/" \
        --batch_size=8 \
        --warmup_steps=200 \
        --benchmark_steps=2000 \
        --use_synthetic_data

    # --------------- #
    '''
  try:
    from tensorflow.python import _pywrap_nvtx as nvtx
  except ImportError:

    class DummyNvtx:

      def __init__(self):
        pass

      def push(self, a=None, b=None):
        pass

      def pop(self, a=None):
        pass

    nvtx = DummyNvtx()

  import os
  import time
  import argparse

  import numpy as np

  os.environ["CUDA_VISIBLE_DEVICES"] = os.environ[
      "CUDA_VISIBLE_DEVICES"] = os.environ.get(
          "CUDA_VISIBLE_DEVICES", str(MPI.COMM_WORLD.Get_rank() % 8))
  #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
  os.environ['TF_GPU_THREAD_COUNT'] = '1'
  os.environ["TF_NUM_INTRAOP_THREADS"]="4"
  os.environ["TF_NUM_INTEROP_THREADS"]="10"

  logging.set_verbosity(logging.INFO)

  parser = argparse.ArgumentParser(description="MaskRCNN Dataloader Benchmark")

  parser.add_argument(
      '--data_dir',
      required=True,
      type=str,
      help="Directory path which contains the preprocessed DAGM 2007 dataset")

  parser.add_argument('--batch_size',
                      default=64,
                      type=int,
                      required=True,
                      help="""Batch size used to measure performance.""")

  parser.add_argument(
      '--warmup_steps',
      default=200,
      type=int,
      required=True,
      help=
      """Number of steps considered as warmup and not taken into account for performance measurements."""
  )

  parser.add_argument(
      '--benchmark_steps',
      default=200,
      type=int,
      required=True,
      help=
      "Number of steps used to benchmark dataloading performance. Only used in training"
  )

  parser.add_argument('--seed',
                      default=666,
                      type=int,
                      required=False,
                      help="""Reproducibility Seed.""")

  parser.add_argument('--log_every',
                      default=1000,
                      type=int,
                      required=False,
                      help="""print a log ever n steps.""")

  parser.add_argument("--training",
                      default=False,
                      action="store_true",
                      help="Benchmark in training mode")

  parser.add_argument("--save_steps",
                      default=False,
                      action="store_true",
                      help="Save individual steps")

  parser.add_argument("--tf2",
                      default=False,
                      action="store_true",
                      help="Run TF2 mode")

  parser.add_argument("--use_synthetic_data",
                      default=False,
                      action="store_true",
                      help="Use synthetic dataset")

  parser.add_argument("--dist_eval",
                      default=False,
                      action="store_true",
                      help="Do distributed evaluation")

  FLAGS, unknown_args = parser.parse_known_args()
  if not FLAGS.tf2:
    tf.compat.v1.disable_eager_execution()

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  if len(unknown_args) > 0:

    for bad_arg in unknown_args:
      print("ERROR: Unknown command line arg: %s" % bad_arg)

    raise ValueError("Invalid command line arg(s)")

  BURNIN_STEPS = FLAGS.warmup_steps

  if FLAGS.training:
    TOTAL_STEPS = FLAGS.warmup_steps + FLAGS.benchmark_steps
  else:
    TOTAL_STEPS = int(1e6)  # Wait for end of dataset

  if FLAGS.training:
    input_dataset = InputReader(file_pattern=os.path.join(
        FLAGS.data_dir, "train*.tfrecord"),
                                mode=tf.estimator.ModeKeys.TRAIN,
                                use_fake_data=FLAGS.use_synthetic_data,
                                use_instance_mask=True,
                                seed=FLAGS.seed)

  else:
    input_dataset = InputReader(file_pattern=os.path.join(
        FLAGS.data_dir, "val*.tfrecord"),
                                mode=tf.estimator.ModeKeys.PREDICT,
                                num_examples=5000,
                                use_fake_data=FLAGS.use_synthetic_data,
                                use_instance_mask=True,
                                seed=FLAGS.seed)

  logging.info("[*] Executing Benchmark in %s mode" %
               ("training" if FLAGS.training else "inference"))
  logging.info("[*] Benchmark using %s data" %
               ("synthetic" if FLAGS.use_synthetic_data else "real"))

  time.sleep(1)

  # Build the data input
  ds_params = {
      "anchor_scale": 8.0,
      "aspect_ratios": [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]],
      "batch_size": FLAGS.batch_size,
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
      "disable_options": False
  }
  ds_params["image_size"] = [832, 1344]

  dataset = input_dataset(params=ds_params)

  if not FLAGS.tf2:
    dataset_iterator = dataset.make_initializable_iterator()

    if FLAGS.training:
      X, Y = dataset_iterator.get_next()
    else:
      X = dataset_iterator.get_next()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False

    with tf.device("gpu:0"):

      X_gpu_ops = list()
      Y_gpu_ops = list()

      if FLAGS.training:

        for _, _x in X.items():
          X_gpu_ops.append(tf.identity(_x))

        for _, _y in Y.items():
          Y_gpu_ops.append(tf.identity(_y))

      else:

        for _, _x in X["features"].items():
          X_gpu_ops.append(tf.identity(_x))

      with tf.control_dependencies(X_gpu_ops + Y_gpu_ops):
        input_op = tf.constant(1.0)

      with tf.compat.v1.Session(config=config) as sess:

        sess.run(dataset_iterator.initializer)

        sess.run(tf.compat.v1.global_variables_initializer())

        total_files_processed = 0

        img_per_sec_arr = []
        processing_time_arr = []

        processing_start_time = time.time()

        for step in range(TOTAL_STEPS):

          try:

            start_time = time.time()
            sess.run(input_op)
            elapsed_time = (time.time() - start_time) * 1000

            imgs_per_sec = (FLAGS.batch_size / elapsed_time) * 1000
            total_files_processed += FLAGS.batch_size

            if (step + 1) > BURNIN_STEPS:
              processing_time_arr.append(elapsed_time)
              img_per_sec_arr.append(imgs_per_sec)

            if (step + 1) % 20 == 0 or (step + 1) == TOTAL_STEPS:
              print(
                  "[STEP %04d] # Batch Size: %03d - Time: %03d msecs - Speed: %6d img/s"
                  % (step + 1, FLAGS.batch_size, elapsed_time, imgs_per_sec))

          except tf.errors.OutOfRangeError:
            break
      processing_time = time.time() - processing_start_time

      avg_processing_speed = np.mean(img_per_sec_arr)

      print(
          "\n###################################################################"
      )
      print("*** Data Loading Performance Metrics ***\n")
      print("\t=> Number of Steps: %d" % (step + 1))
      print("\t=> Batch Size: %d" % FLAGS.batch_size)
      print("\t=> Files Processed: %d" % total_files_processed)
      print("\t=> Total Execution Time: %d secs" % processing_time)
      print("\t=> Median Time per step: %3d msecs" %
            np.median(processing_time_arr))
      print("\t=> Median Processing Speed: %d images/secs" %
            np.median(img_per_sec_arr))
      print("\t=> Median Processing Time: %.2f msecs/image" %
            (1 / float(np.median(img_per_sec_arr)) * 1000))

  else:
    data_iter = iter(dataset)
    total_files_processed = 0
    log_step = FLAGS.log_every
    img_per_sec_arr = []
    processing_time_arr = []
    processing_start_time = time.perf_counter()
    individual_processing_timing_data = []
    warmup_timings = []
    warmup_fps = []
    for step in range(TOTAL_STEPS):
      curr_time = time.perf_counter()
      try:
        r = nvtx.push(f"DataFetch-{step}", f"DataFetch-{step}")
        features,labels = next(data_iter)
        nvtx.pop(r)
      except:
        break
      now = time.perf_counter()
      elapsed_time = now - curr_time  #in seconds
      imgs_per_sec = FLAGS.batch_size / elapsed_time  #in seconds
      total_files_processed += FLAGS.batch_size

      if (step + 1) > BURNIN_STEPS:
        processing_time_arr.append(elapsed_time * 1000.)  # in msec
        img_per_sec_arr.append(imgs_per_sec)
        individual_processing_timing_data.append(
            (elapsed_time * 1000, float(features["source_ids"].numpy()),
             features["image_info"].numpy()))
      else:
        warmup_timings.append(elapsed_time * 1000)
        warmup_fps.append(imgs_per_sec)
      if (step + 1) % log_step == 0 or (step + 1) == TOTAL_STEPS:
        if (step + 1) >= BURNIN_STEPS + log_step:
          print(
              "[STEP %04d] # Batch Size: %03d - Time: %6.3f +/- %6.3f msecs - Speed: %6.3f +/- %6.3f img/s"
              % (step + 1, FLAGS.batch_size,
                 np.mean(processing_time_arr[-log_step:]),
                 np.std(processing_time_arr[-log_step:]),
                 np.mean(img_per_sec_arr[-log_step:]),np.std(img_per_sec_arr[-log_step:])))
        else:
          print(
              "[STEP %04d] # Batch Size: %03d - Time: %06.3f +/- %6.3f msecs - Speed: %6.3f +/- %6.3f img/s"
              %
              (step + 1, FLAGS.batch_size, np.mean(warmup_timings[-log_step:]),
               np.std(warmup_timings[-log_step:]),
               np.mean(warmup_fps[-log_step:]),
               np.mean(warmup_fps[-log_step:])))

    processing_time = time.perf_counter() - processing_start_time
    avg_processing_speed = np.mean(img_per_sec_arr)

    individual_timings = np.array(individual_processing_timing_data)
    if FLAGS.save_steps:
      np.savez("DataPipelineIndividualTimings.npz", timings=individual_timings)
    print(
        "\n###################################################################")
    print("*** Data Loading Performance Metrics ***\n")
    print("\t=> Number of Steps: %s" % (step + 1))
    print("\t=> Batch Size: %s" % FLAGS.batch_size)
    print("\t=> Files Processed: %s" % total_files_processed)
    print("\t=> Total Execution Time: %s secs" % processing_time)
    print("\t=> Median Time per step: %s msecs" %
          np.median(processing_time_arr))
    print("\t=> Average Time per batch: %s +/- %s msecs" %
          (np.mean(processing_time_arr), np.std(processing_time_arr)))
    print("\t=> Median Processing Speed: %s images/secs" %
          np.median(img_per_sec_arr))
    print("\t=> Median Processing Time: %s msecs/image" %
          (1 / float(np.median(img_per_sec_arr)) * 1000))
