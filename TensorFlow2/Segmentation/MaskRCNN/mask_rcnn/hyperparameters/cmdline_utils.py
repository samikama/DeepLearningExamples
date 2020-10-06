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

"""Defining common model params used across all the models."""

from absl import flags


def define_hparams_flags():

    flags.DEFINE_string(
        'log_path',
        default="./mrcnn.json",
        help=(
            'The path where dllogger json file will be saved. Please include the'
            ' name of the json file as well.'
        )
    )

    flags.DEFINE_string(
        'data_dir',
        default=None,
        help=(
            'The directory where the input data is stored. Please see the model'
            ' specific README.md for the expected data format.'
        )
    )

    flags.DEFINE_string('checkpoint', default='', help='Checkpoint filepath')
    
    flags.DEFINE_string('loop_mode', default='estimator', help='Mode to run training estimator, session, or tape.')

    flags.DEFINE_integer(
        'eval_batch_size',
        default=8,
        help='Batch size for evaluation.'
    )

    flags.DEFINE_bool(
        'eval_after_training',
        default=True,
        help='Run one eval after the training finishes.'
    )
    
    flags.DEFINE_integer(
        'first_eval',
        default=0,
        help='Number of early epochs to skip eval.'
    )
    
    flags.DEFINE_integer(
        'log_interval',
        default=10,
        help='How often to print training log.'
    )
    
    flags.DEFINE_integer(
        'loss_scale',
        default=0,
        help='A fixed loss scaling value, if using amp and not set, defaults to dynamic.'
    )
    
    flags.DEFINE_bool(
        'disable_tf2_behavior',
        default=False,
        help='Disable Tensorflow 2 behavior.'
    )

    flags.DEFINE_integer('eval_samples', default=5000, help='Number of training steps')

    flags.DEFINE_bool(
        'include_groundtruth_in_features',
        default=False,
        help=(
            'If `val_json_file` is not provided, one can also read groundtruth'
            ' from input by setting `include_groundtruth_in_features`=True'
        )
    )
    
    flags.DEFINE_bool(
        'dist_eval',
        default=False,
        help=('If set then distribute evaluation amongst all workers'
        )
    )
    
    flags.DEFINE_bool(
        'static_data',
        default=False,
        help=('If set use static dataloader for training (precomputing anchors).')
    )
    
    flags.DEFINE_bool(
        'disable_data_options',
        default=False,
        help=('Disable experimental data options.')
    )
    
    flags.DEFINE_bool(
        'data_slack',
        default=False,
        help=('Enable data prefetching slack.')
    )
    
    flags.DEFINE_bool(
        'tf2',
        default=False,
        help=('Run using TF2 Keras.model'
        )
    )
    
    flags.DEFINE_bool(
        'async_eval',
        default=False,
        help=('asyn evaluation to run during next epoch.'
        )
    )
    
    flags.DEFINE_bool(
        'use_ext',
        default=False,
        help=('use c++ extionsion for fast eval (must have nvidia pycocotools).'
        )
    )
    

    # Gradient clipping is a fairly coarse heuristic to stabilize training.
    # This model clips the gradient by its L2 norm globally (i.e., across
    # all variables), using a threshold obtained from multiplying this
    # parameter with sqrt(number_of_weights), to have a meaningful value
    # across both training phases and different sizes of imported modules.
    # Refer value: 0.02, for 25M weights, yields clip norm 10.
    # Zero or negative number means no clipping.
    flags.DEFINE_float("global_gradient_clip_ratio", default=-1.0, help="Global Gradient Clipping Ratio")

    flags.DEFINE_float("init_learning_rate", default=2.5e-3, help="Initial Learning Rate")

    flags.DEFINE_float("warmup_learning_rate", default=0., help="Warmup Learning Rate Decay Factor")

    flags.DEFINE_bool('finetune_bn', False, 'is batchnorm training mode')

    flags.DEFINE_float("l2_weight_decay", default=1e-4, help="l2 regularization weight")

    flags.DEFINE_string('mode', default='train_and_eval', help='Mode to run: train or eval')
    
    flags.DEFINE_string('optimizer_type', default='SGD', help='Optimizer to use - SGD or Novograd')
    
    flags.DEFINE_string('box_loss_type', default='huber', help='Box loss to use for final box refinement stage - huber or giou')
    
    flags.DEFINE_string('lr_schedule', default='piecewise', help='Learning rate schedule - piecewise or cosine')

    flags.DEFINE_string(
        'model_dir',
        default=None,
        help='The directory where the model and training/evaluation summaries are stored.'
    )

    flags.DEFINE_float("momentum", default=0.9, help="Optimizer Momentum")
    flags.DEFINE_float("beta1", default=0.9, help="novograd b1")
    flags.DEFINE_float("beta2", default=0.3, help="novograd b2")
    flags.DEFINE_integer('num_steps_per_eval', default=2500, help='Number of steps per evaluation epoch.')

    flags.DEFINE_integer('save_checkpoints_steps', default=2500, help='Save a checkpoint every N steps.')

    flags.DEFINE_integer('seed', default=None, help='Set a debug seed for reproducibility.')

    flags.DEFINE_integer('train_batch_size', default=2, help='Batch size for training.')

    flags.DEFINE_integer(
        'total_steps',
        default=938240,
        help=(
            'The number of steps to use for training. This flag'
            ' should be adjusted according to the --train_batch_size flag.'
        )
    )

    flags.DEFINE_list(
        'learning_rate_decay_levels',
        default=['0.1', '0.01'],
        help=(
            'The learning rate decay levels which modify the learning rate using the formula:'
            ' `lr = decay * init_lr`. Decay factor applied at learning_rate_steps.'
        )
    )
    flags.DEFINE_list(
        'learning_rate_steps',
        default=['480000', '640000'],
        help=(
            'The steps at which learning rate changes. This flag'
            ' should be adjusted according to the --train_batch_size flag.'
        )
    )
    flags.DEFINE_integer('warmup_steps', default=1000, help='The number of steps to use warmup learning rate for')

    flags.DEFINE_bool('amp', default=False, help='Enable automatic mixed precision')

    flags.DEFINE_bool(
        'use_batched_nms',
        default=False,
        help='Enable Batched NMS at inference.'
    )

    flags.DEFINE_bool(
        'use_custom_box_proposals_op',
        default=False,
        help='Use GenerateBoundingBoxProposals op.'
    )

    flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')

    flags.DEFINE_bool(
        'use_tf_distributed',
        default=False,
        help='Use tensorflow distributed API'
    )

    flags.DEFINE_bool('xla', default=False, help='Enable XLA JIT Compiler.')

    flags.DEFINE_string('training_file_pattern', default="", help='TFRecords file pattern for the training files')

    flags.DEFINE_string('validation_file_pattern', default="", help='TFRecords file pattern for the validation files')

    flags.DEFINE_string('val_json_file', default="", help='Filepath for the validation json file')

    flags.DEFINE_bool('run_herring', default=False, help='Enable Herring')
    flags.DEFINE_bool('offload_eval', default=False, help='Run eval on another node')
    ############################# TO BE REMOVED ###################################

    flags.DEFINE_integer(
        'report_frequency',
        default=None,
        help='The amount of batches in between accuracy reports at evaluation time'
    )

    ############################# TO BE REMOVED ###################################

    ############################### ISSUES TO FIX - FLAGS #############################"

    # TODO: Remove when XLA at inference fixed
    flags.DEFINE_bool(
        'allow_xla_at_inference',
        default=False,
        help='Enable XLA JIT Compiler at Inference'
    )

    return flags.FLAGS
