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
from collections import deque
import os
import cProfile, pstats

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'
# os.environ["TF_XLA_FLAGS"] = 'tf_xla_print_cluster_outputs=1'

from absl import app

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

#from tensorflow.keras.mixed_precision import experimental as mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)
#Remeber to verify this runs on the OFFLOAD, Can run in Main file
#tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
#tf.config.optimizer.set_jit(True)

from mask_rcnn.utils.logging_formatter import logging
from mask_rcnn.utils.distributed_utils import MPI_is_distributed

from mask_rcnn import dataloader
from mask_rcnn import distributed_executer
from mask_rcnn import mask_rcnn_model as mask_rcnn_model_v1
from mask_rcnn.tf2 import mask_rcnn_model as mask_rcnn_model_v2
from mask_rcnn import session_executor
from mask_rcnn import tape_executor
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn.hyperparameters import params_io
from mask_rcnn.tf2.mask_rcnn_model import TapeModel
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

import h5py
import time
import multiprocessing as mp




def get_latest_checkpoint(q, checkpoint_dir, model):
    encountered_checkpoints = set() 
    while True:
        latest=model.get_latest_checkpoint()
        if (latest is not None) and (len(q) == 0 or q[-1] != latest) and (latest not in encountered_checkpoints):
            q.append(latest)
            encountered_checkpoints.add(latest)
        time.sleep(1)




def do_eval(run_config, train_input_fn, eval_input_fn):

    mrcnn_model = TapeModel(run_config, train_input_fn, eval_input_fn, is_training=True)
    q = deque()
    out_path = run_config.model_dir
    args=[q, out_path, mrcnn_model]
    eval_workers=min(32, MPI_size())
    batches = []
    total_samples = 0
    img_ids = []
    while 1:
      try:
        batches.append(next(mrcnn_model.eval_tdf)['features'])
        total_samples += batches[-1]['images'].shape[0]
        #print(batches[-1]['images'].shape[0])
        img_ids += list(batches[-1]['source_ids'].numpy().flatten())

      except Exception as e:
        #Should only break when out of data
        break
    steps = len(batches)
    print(total_samples)

    print(len(set(img_ids)))
    
    comm = MPI.COMM_WORLD
    res = comm.gather(img_ids)
    if(MPI_rank() == 0):
      res = sum(res, [])
      print("Total imgs ", len(set(res)))

    for ii in range(len(batches)):
      tmpdict = {}
      for key in batches[ii]:
        tmpdict[key] = batches[ii][key].numpy()
      batches[ii] = tmpdict
    # import pickle
    # with open("/tmp/input_b37", 'wb') as fp:
    #   pickle.dump(batches, fp)
    
    mrcnn_model.initialize_eval_model(batches[0])
   
    #if MPI_rank() == 0:
    chkpoint_thread = threading.Thread(target=get_latest_checkpoint, name="checkpoint thread", args=args)
    chkpoint_thread.start()
    last = None
    test = True
    
    while True:
        if len(q) == 0 or q[0] == last:
            pass
        if len(q) !=0 and q[0] != last:
            last = q[0]
            q.popleft()
            print("#"*20, "Running eval for", last)
            mrcnn_model.load_model(os.path.join(run_config.model_dir,last))
            mrcnn_model.run_eval(steps, batches, async_eval=run_config.async_eval,
                    use_ext=run_config.use_ext, use_dist_coco_eval=run_config.dist_coco_eval)
        time.sleep(5)

def main(argv):
    del argv  # Unused.

    # ============================ Configure parameters ============================ #
    RUN_CONFIG = mask_rcnn_params.default_config()

    temp_config = FLAGS.flag_values_dict()
    for i,j in temp_config.items():
      if MPI_rank() == 0:
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
