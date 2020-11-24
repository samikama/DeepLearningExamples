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

import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'
# os.environ["TF_XLA_FLAGS"] = 'tf_xla_print_cluster_outputs=1'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'
os.environ["TF_NUM_INTRAOP_THREADS"]="7"
os.environ["TF_NUM_INTEROP_THREADS"]="9"
os.environ["ENABLE_NVTX_MARKERS"]="0"
# TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_XXX_OOO where XXX is ["WHITELIST","GRAYLIST","BLACKLIST","CLEARLIST"] and OOO is ["ADD", "REMOVE"]
# list is op name with comma separators
#os.environ["TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_ADD"]= "BatchedBoxProposals,ROIAlign,ROIAlignGrad"
from absl import app

import tensorflow as tf
#from tensorflow.python.framework.ops import disable_eager_execution

# tf.config.run_functions_eagerly(True)

from mask_rcnn.utils.herring_env import is_herring

if is_herring():
    import herring.tensorflow as herring
    herring.init()

    from mask_rcnn.utils.distributed_utils_herring import MPI_rank, MPI_is_distributed
else:
    from mpi4py import MPI
    CURR_GPU_INDEX=str(MPI.COMM_WORLD.Get_rank()%8)
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES",CURR_GPU_INDEX)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'    
    from mask_rcnn.utils.distributed_utils import MPI_rank, MPI_is_distributed



from mask_rcnn.utils.logging_formatter import logging

from mask_rcnn import dataloader
from mask_rcnn.tf2 import mask_rcnn_model as mask_rcnn_model_v2

from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn.hyperparameters import params_io

from mask_rcnn.hyperparameters.cmdline_utils import define_hparams_flags

from mask_rcnn.utils.logging_formatter import log_cleaning
#import dllogger

FLAGS = define_hparams_flags()

def run_executer(runtime_config, train_input_fn=None, eval_input_fn=None):
    """Runs Mask RCNN model on distribution strategy defined by the user."""
    from mask_rcnn import distributed_executer
    from mask_rcnn import mask_rcnn_model as mask_rcnn_model_v1
    mask_rcnn_model = mask_rcnn_model_v2 if runtime_config.tf2 else mask_rcnn_model_v1
    if runtime_config.use_tf_distributed:
        executer = distributed_executer.TFDistributedExecuter(runtime_config, mask_rcnn_model.mask_rcnn_model_fn)
    else:
        executer = distributed_executer.EstimatorExecuter(runtime_config, mask_rcnn_model.mask_rcnn_model_fn)

    if runtime_config.mode == 'train':
        executer.train(
            train_input_fn=train_input_fn,
            run_eval_after_train=FLAGS.eval_after_training,
            eval_input_fn=eval_input_fn
        )

    elif runtime_config.mode == 'eval':
        executer.eval(eval_input_fn=eval_input_fn)

    elif runtime_config.mode == 'train_and_eval':
        executer.train_and_eval(train_input_fn=train_input_fn, eval_input_fn=eval_input_fn)

    else:
        raise ValueError('Mode must be one of `train`, `eval`, or `train_and_eval`')
        
def run_session(runtime_config, train_input_fn, eval_input_fn):
    from mask_rcnn import session_executor
    session_executor.train_and_eval(runtime_config, train_input_fn, eval_input_fn)

def run_tape(runtime_config, train_input_fn, eval_input_fn):
    from mask_rcnn import tape_executor
    tape_executor.train_and_eval(runtime_config, train_input_fn, eval_input_fn)

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
    if RUN_CONFIG.disable_tf2_behavior:
        tf.compat.v1.disable_v2_behavior()
    
    # ============================ Configure parameters ============================ #


    
    if RUN_CONFIG.use_tf_distributed and MPI_is_distributed():
        raise RuntimeError("Incompatible Runtime. Impossible to use `--use_tf_distributed` with MPIRun Horovod")

    if RUN_CONFIG.mode in ('train', 'train_and_eval') and not RUN_CONFIG.training_file_pattern:
        raise RuntimeError('You must specify `training_file_pattern` for training.')

    if RUN_CONFIG.mode in ('eval', 'train_and_eval'):
        if not RUN_CONFIG.validation_file_pattern:
            raise RuntimeError('You must specify `validation_file_pattern` for evaluation.')

        if RUN_CONFIG.val_json_file == "" and not RUN_CONFIG.include_groundtruth_in_features:
            raise RuntimeError(
                'You must specify `val_json_file` or include_groundtruth_in_features=True for evaluation.')

        if not RUN_CONFIG.include_groundtruth_in_features and not os.path.isfile(RUN_CONFIG.val_json_file):
            raise FileNotFoundError("Validation JSON File not found: %s" % RUN_CONFIG.val_json_file)

    #dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
    #                                                       filename=RUN_CONFIG.log_path)])

    if RUN_CONFIG.mode in ('train', 'train_and_eval'):
        
        if RUN_CONFIG.static_data:
            train_input_fn = static_data.FastDataLoader(RUN_CONFIG.training_file_pattern, 
                                                        RUN_CONFIG.values())
        else:
            train_input_fn = dataloader.InputReader(
                file_pattern=RUN_CONFIG.training_file_pattern,
                mode=tf.estimator.ModeKeys.TRAIN,
                num_examples=None,
                use_fake_data=RUN_CONFIG.use_fake_data,
                use_instance_mask=RUN_CONFIG.include_mask,
                seed=RUN_CONFIG.seed,
                disable_options=RUN_CONFIG.disable_data_options
            )

    else:
        train_input_fn = None

    if RUN_CONFIG.mode in ('eval', 'train_and_eval' or (RUN_CONFIG.mode == 'train' and RUN_CONFIG.eval_after_training)):

        eval_input_fn = dataloader.InputReader(
            file_pattern=RUN_CONFIG.validation_file_pattern,
            mode=tf.estimator.ModeKeys.PREDICT,
            num_examples=RUN_CONFIG.eval_samples,
            use_fake_data=False,
            use_instance_mask=RUN_CONFIG.include_mask,
            seed=RUN_CONFIG.seed,
            disable_options=RUN_CONFIG.disable_data_options
        )

    else:
        eval_input_fn = None
        
    if RUN_CONFIG.loop_mode=='estimator':
        run_executer(RUN_CONFIG, train_input_fn, eval_input_fn)
    elif RUN_CONFIG.loop_mode=='session':
        run_session(RUN_CONFIG, train_input_fn, eval_input_fn)
    elif RUN_CONFIG.loop_mode=='tape':
        run_tape(RUN_CONFIG, train_input_fn, eval_input_fn)

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    
    logging.set_verbosity(logging.DEBUG)
    tf.autograph.set_verbosity(0)
    log_cleaning(hide_deprecation_warnings=True)
    app.run(main)
