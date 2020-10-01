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

"""Training script for Mask-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'
# os.environ["TF_XLA_FLAGS"] = 'tf_xla_print_cluster_outputs=1'

from absl import app

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

from mask_rcnn.utils.logging_formatter import logging
from mask_rcnn.utils.distributed_utils import MPI_is_distributed

from mask_rcnn import dataloader
from mask_rcnn import distributed_executer
from mask_rcnn import mask_rcnn_model as mask_rcnn_model_v1
from mask_rcnn.tf2 import mask_rcnn_model as mask_rcnn_model_v2
from mask_rcnn import session_executor
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn.hyperparameters import params_io

from mask_rcnn.hyperparameters.cmdline_utils import define_hparams_flags

from mask_rcnn.utils.logging_formatter import log_cleaning
import dllogger

FLAGS = define_hparams_flags()


import os
import sys
from statistics import mean
import copy
import threading
from math import ceil

import numpy as np
from tqdm import tqdm
from mask_rcnn.utils.distributed_utils import MPI_is_distributed, MPI_rank, MPI_size, MPI_local_rank
from mpi4py import MPI
import horovod.tensorflow as hvd
hvd.init()

from mask_rcnn.tf2.mask_rcnn_model import SessionModel
from mask_rcnn.hooks import pretrained_restore_hook
from mask_rcnn import evaluation

from mask_rcnn.training import losses, learning_rates
from mask_rcnn import coco_metric
from mask_rcnn.utils import coco_utils

import time

def run_eval(model, sess, steps, params, async_eval=True, use_ext=True):
    if MPI_rank()==0:
        p_bar = tqdm(range(steps), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        logging.info("Starting eval loop")
    else:
        p_bar = range(steps)
    worker_predictions = dict()
    for i in p_bar:
        out = sess.run(model.eval_step)
        out = evaluation.process_prediction_for_eval(out)
        for k, v in out.items():
            if k not in worker_predictions:
                worker_predictions[k] = [v]
            else:
                worker_predictions[k].append(v)
    if MPI_rank()==0:
        logging.info("Processing eval")
    coco = coco_metric.MaskCOCO()
    _preds = copy.deepcopy(worker_predictions)
    for k, v in _preds.items():
        _preds[k] = np.concatenate(v, axis=0)
    #if MPI_rank() < 32:
    converted_predictions = coco.load_predictions(_preds, include_mask=True, is_image_mask=False)
    worker_source_ids = _preds['source_id']
    MPI.COMM_WORLD.barrier()
    predictions_list = evaluation.gather_result_from_all_processes(converted_predictions)
    source_ids_list = evaluation.gather_result_from_all_processes(worker_source_ids)
    validation_json_file=params.val_json_file
    if MPI_rank() == 0:
        all_predictions = []
        source_ids = []
        for i, p in enumerate(predictions_list):
            all_predictions.extend(p)
        for i, s in enumerate(source_ids_list):
            source_ids.extend(s)
        if use_ext:
            args = [all_predictions, validation_json_file]
            if async_eval:
                
                eval_thread = threading.Thread(target=evaluation.fast_eval,
                                                   name="eval-thread", args=args)
                eval_thread.start()
            else:
                evaluation.fast_eval(*args)
        else:
            args = [all_predictions, source_ids, True, validation_json_file]
            if async_eval:
                eval_thread = threading.Thread(target=evaluation.compute_coco_eval_metric_n, name="eval-thread", args=args)
                eval_thread.start()
            else:
                evaluation.compute_coco_eval_metric_n(*args)

def run_eval_old(model, sess, steps, params, async_eval=False):
    if MPI_rank()==0:
        p_bar = tqdm(range(steps), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        logging.info("Starting eval loop")
    else:
        p_bar = range(steps)
    worker_predictions = dict()
    for i in p_bar:
        out = sess.run(model.eval_step)
        out = evaluation.process_prediction_for_eval(out)
        for k, v in out.items():
            if k not in worker_predictions:
                worker_predictions[k] = [v]
            else:
                worker_predictions[k].append(v)
    coco = coco_metric.MaskCOCO()
    _preds = copy.deepcopy(worker_predictions)
    for k, v in _preds.items():
        _preds[k] = np.concatenate(v, axis=0)
    if MPI_rank() < 32:
        converted_predictions = coco.load_predictions(_preds, include_mask=True, is_image_mask=False)
        worker_source_ids = _preds['source_id']
    else:
        converted_predictions = []
        worker_source_ids = []
    MPI.COMM_WORLD.barrier()
    predictions_list = evaluation.gather_result_from_all_processes(converted_predictions)
    source_ids_list = evaluation.gather_result_from_all_processes(worker_source_ids)
    validation_json_file=params.val_json_file
    if MPI_rank() == 0:
        all_predictions = []
        source_ids = []
        for i, p in enumerate(predictions_list):
            if i < 32:
                all_predictions.extend(p)
        for i, s in enumerate(source_ids_list):
            if i < 32:
                source_ids.extend(s)
        args = [all_predictions, source_ids, True, validation_json_file]
        if async_eval:
            eval_thread = threading.Thread(target=evaluation.compute_coco_eval_metric_n, name="eval-thread", args=args)
            eval_thread.start()
        else:
            evaluation.compute_coco_eval_metric_n(*args)

def do_eval(run_config, train_input_fn, eval_input_fn):
    model = SessionModel(run_config, train_input_fn, eval_input_fn)
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    out_path = '/home/ubuntu/fsx/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/results_session_1x/'
    last = out_path+'model.ckpt-0'
    saver = tf.compat.v1.train.Saver()
    sess_config = model.get_session_config(use_xla=run_config.xla)
    session_creator=tf.compat.v1.train.ChiefSessionCreator(config=sess_config)
    sess = tf.compat.v1.train.MonitoredSession(session_creator=session_creator, hooks=hooks)

    eval_workers = min(32, MPI_size())
    latest = tf.train.latest_checkpoint("/home/ubuntu/fsx/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/results_session_1x/")
    
    while True:
        if last != latest and latest is not None:
            print("#"*20, "New latest found", latest)
            last = latest
            sess.run(model.eval_tdf.initializer)
            saver.restore(sess,latest)
            run_eval(model, sess, run_config.eval_samples//eval_workers, run_config)
        else:
            latest = tf.train.latest_checkpoint("/home/ubuntu/fsx/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/results_session_1x/")
            print("#"*20,"Nothing new here")
            time.sleep(5)

def main(argv):
    del argv  # Unused.

    # ============================ Configure parameters ============================ #
    RUN_CONFIG = mask_rcnn_params.default_config()

    temp_config = FLAGS.flag_values_dict()
    for i,j in temp_config.items():
        print("{}: {}".format(i,j))
    temp_config['learning_rate_decay_levels'] = [float(decay) for decay in temp_config['learning_rate_decay_levels']]
    temp_config['learning_rate_levels'] = [
        decay * temp_config['init_learning_rate'] for decay in temp_config['learning_rate_decay_levels']
    ]
    temp_config['learning_rate_steps'] = [int(step) for step in temp_config['learning_rate_steps']]

    RUN_CONFIG = params_io.override_hparams(RUN_CONFIG, temp_config)
    
    if RUN_CONFIG.loop_mode in ['estimator', 'session']:
        disable_eager_execution()
    if RUN_CONFIG.loop_mode=='session':
        tf.compat.v1.disable_v2_behavior()
    
    # ============================ Configure parameters ============================ #

    if RUN_CONFIG.use_tf_distributed and MPI_is_distributed():
        raise RuntimeError("Incompatible Runtime. Impossible to use `--use_tf_distributed` with MPIRun Horovod")


    if RUN_CONFIG.mode in ('eval', 'train_and_eval'):
        if not RUN_CONFIG.validation_file_pattern:
            raise RuntimeError('You must specify `validation_file_pattern` for evaluation.')

        if RUN_CONFIG.val_json_file == "" and not RUN_CONFIG.include_groundtruth_in_features:
            raise RuntimeError(
                'You must specify `val_json_file` or include_groundtruth_in_features=True for evaluation.')

        if not RUN_CONFIG.include_groundtruth_in_features and not os.path.isfile(RUN_CONFIG.val_json_file):
            raise FileNotFoundError("Validation JSON File not found: %s" % RUN_CONFIG.val_json_file)

    dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=RUN_CONFIG.log_path)])

    eval_input_fn = dataloader.InputReader(
        file_pattern=RUN_CONFIG.validation_file_pattern,
        mode=tf.estimator.ModeKeys.PREDICT,
        num_examples=RUN_CONFIG.eval_samples,
        use_fake_data=False,
        use_instance_mask=RUN_CONFIG.include_mask,
        seed=RUN_CONFIG.seed,
        disable_options=RUN_CONFIG.disable_data_options
    )
    
    train_input_fn = dataloader.InputReader(
        file_pattern=RUN_CONFIG.training_file_pattern,
        mode=tf.estimator.ModeKeys.TRAIN,
        num_examples=None,
        use_fake_data=RUN_CONFIG.use_fake_data,
        use_instance_mask=RUN_CONFIG.include_mask,
        seed=RUN_CONFIG.seed,
        disable_options=RUN_CONFIG.disable_data_options
    )    
    do_eval(RUN_CONFIG, train_input_fn, eval_input_fn)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    
    logging.set_verbosity(logging.DEBUG)
    tf.autograph.set_verbosity(0)
    log_cleaning(hide_deprecation_warnings=True)

    app.run(main)
