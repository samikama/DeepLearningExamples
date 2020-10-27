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

"""Model definition for the Mask-RCNN Model.

Defines model_fn of Mask-RCNN for TF Estimator. The model_fn includes Mask-RCNN
model architecture, loss function, learning rate schedule, and evaluation
procedure.

"""
import time
import itertools
import copy
import numpy as np
from statistics import mean
import threading
from math import ceil
from mpi4py import MPI
from tqdm import tqdm
import os

import multiprocessing.dummy as mp
import queue
#mp.set_start_method('spawn')

import h5py
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.core.protobuf import rewriter_config_pb2
from mask_rcnn import anchors

from mask_rcnn.models import fpn
from mask_rcnn.models import heads
from mask_rcnn.tf2.models import heads as tf2_heads
from mask_rcnn.models import resnet

from mask_rcnn.training import losses, learning_rates, optimizers

from mask_rcnn.ops import postprocess_ops
from mask_rcnn.ops import roi_ops
from mask_rcnn.ops import spatial_transform_ops
from mask_rcnn.ops import training_ops

from mask_rcnn.utils.logging_formatter import logging

from mask_rcnn.utils.distributed_utils import MPI_is_distributed, MPI_local_rank, MPI_rank
from mask_rcnn import evaluation, coco_metric

from mask_rcnn.utils.meters import StandardMeter
from mask_rcnn.utils.metric_tracking import register_metric

from mask_rcnn.utils.lazy_imports import LazyImport
from mask_rcnn.training.optimization import LambOptimizer, NovoGrad
from mask_rcnn.tf2.utils import warmup_scheduler, eager_mapping

from mask_rcnn.utils.meters import StandardMeter
from mask_rcnn.utils.metric_tracking import register_metric
from mask_rcnn.utils.herring_env import is_herring
from tensorflow.python.profiler import profiler_v2 as tf_profiler
from tensorflow.python.profiler.trace import Trace as prof_Trace
try:
    from tensorflow.python import _pywrap_nvtx as nvtx
except ImportError:
    class DummyNvtx:
        def __init__(self):
            pass
        def push(self,a=None,b=None):
            pass
        def pop(self,a=None):
            pass
    nvtx=DummyNvtx()

if is_herring():
    import herring.tensorflow as herring
else:
    hvd = LazyImport("horovod.tensorflow")

MODELS = dict()

import cProfile, pstats

def profile_dec(func):
  def wrapper(*args, **kwargs):
    with cProfile.Profile() as pr:
      ret = func(*args, **kwargs)
      ps = pstats.Stats(pr).sort_stats('cumtime')
      ps.print_stats()
    return ret
  
  return wrapper

def create_optimizer(learning_rate, params):
    """Creates optimized based on the specified flags."""

    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=params['momentum'])

    if MPI_is_distributed():
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            name=None,
            device_dense='/gpu:0',
            device_sparse='',
            # compression=hvd.Compression.fp16,
            compression=hvd.Compression.none,
            sparse_as_dense=False
        )

    if params["amp"]:
        loss_scale = tf.train.experimental.DynamicLossScale(
            initial_loss_scale=(2 ** 12),
            increment_period=2000,
            multiplier=2.0
        )
        if params['loss_scale']>0:
            optimizer = tf.compat.v1.train.experimental.MixedPrecisionLossScaleOptimizer(optimizer, 
                                                                                         loss_scale=params['loss_scale'])
        else:
            optimizer = tf.compat.v1.train.experimental.MixedPrecisionLossScaleOptimizer(optimizer, loss_scale=loss_scale)

    return optimizer

def create_lamb_optimizer(learning_rate, params):
    """Creates optimized based on the specified flags."""
    optimizer = LAMBOptimizer(
            learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            weight_decay_rate=params['l2_weight_decay'],
            exclude_from_weight_decay=['bias', 'beta', 'batch_normalization']
    )

#    optimizer = tfa.optimizers.LAMB(
#        learning_rate=4e-3,
#        weight_decay_rate=1e-4,
#        beta_1=0.9,
#        beta_2=0.999,
#        epsilon=1e-6,
#        exclude_from_weight_decay=["bias"],
#    )

    if MPI_is_distributed():
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            name=None,
            device_dense='/gpu:0',
            device_sparse='',
            # compression=hvd.Compression.fp16,
            compression=hvd.Compression.none,
            sparse_as_dense=False
        )

    if params["amp"]:
        loss_scale = tf.train.experimental.DynamicLossScale(
            initial_loss_scale=(2 ** 11),
            increment_period=2000,
            multiplier=2.0
        )
        optimizer = tf.compat.v1.train.experimental.MixedPrecisionLossScaleOptimizer(optimizer, loss_scale=loss_scale)

    return optimizer


def create_novograd_optimizer(learning_rate, params):
    """Creates optimized based on the specified flags."""
    optimizer = NovoGrad(
            learning_rate,
            beta_1=0.9,
            beta_2=0.4, #0.5, #0.8, #0.98,
            weight_decay=params['l2_weight_decay'],
            exclude_from_weight_decay=['bias', 'beta', 'batch_normalization']
    )


#    optimizer = tfa.optimizers.NovoGrad(
#        lr=1e-3,
#        beta_1=0.9,
#        beta_2=0.999,
#        weight_decay=0.001,
#        grad_averaging=False
#    )

    if MPI_is_distributed():
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            name=None,
            device_dense='/gpu:0',
            device_sparse='',
            # compression=hvd.Compression.fp16,
            compression=hvd.Compression.none,
            sparse_as_dense=False
        )

    if params["amp"]:
        loss_scale = tf.train.experimental.DynamicLossScale(
            initial_loss_scale=(2 ** 11),
            increment_period=2000,
            multiplier=2.0
        )
        optimizer = tf.compat.v1.train.experimental.MixedPrecisionLossScaleOptimizer(optimizer, loss_scale=loss_scale)

    return optimizer

def compute_model_statistics(batch_size, is_training=True):
    """Compute number of parameters and FLOPS."""
    options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    options['output'] = 'none'

    from tensorflow.python.keras.backend import get_graph
    flops = tf.compat.v1.profiler.profile(get_graph(), options=options).total_float_ops
    flops_per_image = flops / batch_size

    logging.info('[%s Compute Statistics] %.1f GFLOPS/image' % (
        "Training" if is_training else "Inference",
        flops_per_image/1e9
    ))
    
class MRCNN(tf.keras.Model):
    
    def __init__(self, params, is_training=True, **kwargs):
        super().__init__(**kwargs)
        is_gpu_inference = not is_training and params['use_batched_nms']
        self.backbone = resnet.Resnet_Model(
                                        "resnet50",
                                        data_format='channels_last',
                                        trainable=is_training,
                                        finetune_bn=params['finetune_bn']
                                    )
        self.fpn = fpn.FPNNetwork(params['min_level'], 
                                  params['max_level'], 
                                  trainable=is_training)
        self.rpn = heads.RPN_Head_Model(name="rpn_head", 
                                        num_anchors=len(params['aspect_ratios'] * params['num_scales']), 
                                        trainable=is_training)
        self.box_head = heads.Box_Head_Model(
                                    num_classes=params['num_classes'],
                                    mlp_head_dim=params['fast_rcnn_mlp_head_dim'],
                                    trainable=is_training
                                )
        self.mask_head = tf2_heads.Mask_Head_Model(
                                                num_classes=params['num_classes'],
                                                mrcnn_resolution=params['mrcnn_resolution'],
                                                is_gpu_inference=is_gpu_inference,
                                                trainable=is_training,
                                                name="mask_head"
                                            )
    def call(self, features, labels, params, is_training=True):
        model_outputs = {}
        is_gpu_inference = not is_training and params['use_batched_nms']
        batch_size, image_height, image_width, _ = features['images'].get_shape().as_list()
        if 'source_ids' not in features:
            features['source_ids'] = -1 * tf.ones([batch_size], dtype=tf.float32)

        all_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                      params['num_scales'], params['aspect_ratios'],
                                      params['anchor_scale'],
                                      (image_height, image_width))
        backbone_feats = self.backbone(
            features['images'],
            training=is_training,
        )
        fpn_feats = self.fpn(backbone_feats, training=is_training)
        rpn_score_outputs, rpn_box_outputs = self.rpn_head_fn(
                                                    features=fpn_feats,
                                                    min_level=params['min_level'],
                                                    max_level=params['max_level'],
                                                    is_training=is_training)
        if is_training:
            rpn_pre_nms_topn = params['train_rpn_pre_nms_topn']
            rpn_post_nms_topn = params['train_rpn_post_nms_topn']
            rpn_nms_threshold = params['train_rpn_nms_threshold']

        else:
            rpn_pre_nms_topn = params['test_rpn_pre_nms_topn']
            rpn_post_nms_topn = params['test_rpn_post_nms_topn']
            rpn_nms_threshold = params['test_rpn_nms_thresh']

        if params['use_custom_box_proposals_op']:
            rpn_box_scores, rpn_box_rois = roi_ops.custom_multilevel_propose_rois(
                scores_outputs=rpn_score_outputs,
                box_outputs=rpn_box_outputs,
                all_anchors=all_anchors,
                image_info=features['image_info'],
                rpn_pre_nms_topn=rpn_pre_nms_topn,
                rpn_post_nms_topn=rpn_post_nms_topn,
                rpn_nms_threshold=rpn_nms_threshold,
                rpn_min_size=params['rpn_min_size']
            )
        else:
            rpn_box_scores, rpn_box_rois = roi_ops.multilevel_propose_rois(
                scores_outputs=rpn_score_outputs,
                box_outputs=rpn_box_outputs,
                all_anchors=all_anchors,
                image_info=features['image_info'],
                rpn_pre_nms_topn=rpn_pre_nms_topn,
                rpn_post_nms_topn=rpn_post_nms_topn,
                rpn_nms_threshold=rpn_nms_threshold,
                rpn_min_size=params['rpn_min_size'],
                bbox_reg_weights=None,
                use_batched_nms=params['use_batched_nms']
            )
        rpn_box_rois = tf.cast(rpn_box_rois, dtype=tf.float32)

        if is_training:
            rpn_box_rois = tf.stop_gradient(rpn_box_rois)
            rpn_box_scores = tf.stop_gradient(rpn_box_scores)  # TODO Jonathan: Unused => Shall keep ?

            # Sampling
            box_targets, class_targets, rpn_box_rois, proposal_to_label_map = \
            training_ops.proposal_label_op(
                rpn_box_rois,
                labels['gt_boxes'],
                labels['gt_classes'],
                batch_size_per_im=params['batch_size_per_im'],
                fg_fraction=params['fg_fraction'],
                fg_thresh=params['fg_thresh'],
                bg_thresh_hi=params['bg_thresh_hi'],
                bg_thresh_lo=params['bg_thresh_lo']
            )

        # Performs multi-level RoIAlign.
        box_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats,
            boxes=rpn_box_rois,
            output_size=7,
            is_gpu_inference=is_gpu_inference
        )
        
        class_outputs, box_outputs, _ = self.box_head(inputs=box_roi_features)

        if not is_training:
            if params['use_batched_nms']:
                generate_detections_fn = postprocess_ops.generate_detections_gpu

            else:
                generate_detections_fn = postprocess_ops.generate_detections_tpu
            
            detections = generate_detections_fn(
                class_outputs=class_outputs,
                box_outputs=box_outputs,
                anchor_boxes=rpn_box_rois,
                image_info=features['image_info'],
                pre_nms_num_detections=params['test_rpn_post_nms_topn'],
                post_nms_num_detections=params['test_detections_per_image'],
                nms_threshold=params['test_nms'],
                bbox_reg_weights=params['bbox_reg_weights']
            )

            model_outputs.update({
                'num_detections': detections[0],
                'detection_boxes': detections[1],
                'detection_classes': detections[2],
                'detection_scores': detections[3],
            })
            # testing outputs
            #model_outputs.update({'class_outputs': tf.nn.softmax(class_outputs),
            #                      'box_outputs': box_outputs,
            #                      'anchor_boxes': rpn_box_rois})
        else:  # is training
            if params['box_loss_type'] != "giou":
                encoded_box_targets = training_ops.encode_box_targets(
                    boxes=rpn_box_rois,
                    gt_boxes=box_targets,
                    gt_labels=class_targets,
                    bbox_reg_weights=params['bbox_reg_weights']
                )

            model_outputs.update({
                'rpn_score_outputs': rpn_score_outputs,
                'rpn_box_outputs': rpn_box_outputs,
                'class_outputs': class_outputs,
                'box_outputs': box_outputs,
                'class_targets': class_targets,
                'box_targets': encoded_box_targets if params['box_loss_type'] != 'giou' else box_targets,
                'box_rois': rpn_box_rois,
            })
        # Faster-RCNN mode.
        if not params['include_mask']:
            return model_outputs

        # Mask sampling
        if not is_training:
            selected_box_rois = model_outputs['detection_boxes']
            class_indices = model_outputs['detection_classes']
            class_indices = tf.cast(class_indices, dtype=tf.int32)

        else:
            selected_class_targets, selected_box_targets, \
            selected_box_rois, proposal_to_label_map = training_ops.select_fg_for_masks(
                class_targets=class_targets,
                box_targets=box_targets,
                boxes=rpn_box_rois,
                proposal_to_label_map=proposal_to_label_map,
                max_num_fg=int(params['batch_size_per_im'] * params['fg_fraction'])
            )

            class_indices = tf.cast(selected_class_targets, dtype=tf.int32)

        mask_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats,
            boxes=selected_box_rois,
            output_size=14,
            is_gpu_inference=is_gpu_inference
        )
        
        mask_outputs = self.mask_head(inputs=mask_roi_features, class_indices=class_indices)

        '''if MPI_local_rank() == 0:
            # Print #FLOPs in model.
            compute_model_statistics(batch_size, is_training=is_training)'''

        if is_training:
            mask_targets = training_ops.get_mask_targets(

                fg_boxes=selected_box_rois,
                fg_proposal_to_label_map=proposal_to_label_map,
                fg_box_targets=selected_box_targets,
                mask_gt_labels=labels['cropped_gt_masks'],
                output_size=params['mrcnn_resolution']
            )

            model_outputs.update({
                'mask_outputs': mask_outputs,
                'mask_targets': mask_targets,
                'selected_class_targets': selected_class_targets,
            })

        else:
            model_outputs.update({
                'detection_masks': tf.nn.sigmoid(mask_outputs),
            })

        return model_outputs
    
    def rpn_head_fn(self, features, min_level=2, max_level=6, is_training=True):
        scores_outputs = dict()
        box_outputs = dict()
        for level in range(min_level, max_level + 1):
            scores_outputs[level], box_outputs[level] = self.rpn(features[level],
                                                                 training=is_training)
        return scores_outputs, box_outputs
    
def _model_fn(features, labels, mode, params):
    """Model defination for the Mask-RCNN model based on ResNet.

    Args:
    features: the input image tensor and auxiliary information, such as
      `image_info` and `source_ids`. The image tensor has a shape of
      [batch_size, height, width, 3]. The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include score targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.
    """

    # Set up training loss and learning rate.
    global_step = tf.compat.v1.train.get_or_create_global_step()

    if mode == tf.estimator.ModeKeys.PREDICT:

        if params['include_groundtruth_in_features'] and 'labels' in features:
            # In include groundtruth for eval.
            labels = features['labels']

        else:
            labels = None

        if 'features' in features:
            features = features['features']
            # Otherwise, it is in export mode, the features is past in directly.
    model = MRCNN(params)
    model_outputs = model(features, labels, params, mode==tf.estimator.ModeKeys.TRAIN)

    model_outputs.update({
        'source_id': features['source_ids'],
        'image_info': features['image_info'],
    })

    if mode == tf.estimator.ModeKeys.PREDICT and 'orig_images' in features:
        model_outputs['orig_images'] = features['orig_images']

    # First check if it is in PREDICT mode or EVAL mode to fill out predictions.
    # Predictions are used during the eval step to generate metrics.
    if mode in [tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]:
        predictions = {}

        try:
            model_outputs['orig_images'] = features['orig_images']
        except KeyError:
            pass

        if labels and params['include_groundtruth_in_features']:
            # Labels can only be embedded in predictions. The prediction cannot output
            # dictionary as a value.
            predictions.update(labels)

        model_outputs.pop('fpn_features', None)
        predictions.update(model_outputs)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # If we are doing PREDICT, we can return here.
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # score_loss and box_loss are for logging. only total_loss is optimized.
    total_rpn_loss, rpn_score_loss, rpn_box_loss = losses.rpn_loss(
        score_outputs=model_outputs['rpn_score_outputs'],
        box_outputs=model_outputs['rpn_box_outputs'],
        labels=labels,
        params=params
    )

    total_fast_rcnn_loss, fast_rcnn_class_loss, fast_rcnn_box_loss = losses.fast_rcnn_loss(
        class_outputs=model_outputs['class_outputs'],
        box_outputs=model_outputs['box_outputs'],
        class_targets=model_outputs['class_targets'],
        box_targets=model_outputs['box_targets'],
        rpn_box_rois=model_outputs['box_rois'],
        image_info=features['image_info'],
        params=params
    )

    # Only training has the mask loss.
    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/model_builder.py
    if mode == tf.estimator.ModeKeys.TRAIN and params['include_mask']:
        mask_loss = losses.mask_rcnn_loss(
            mask_outputs=model_outputs['mask_outputs'],
            mask_targets=model_outputs['mask_targets'],
            select_class_targets=model_outputs['selected_class_targets'],
            params=params
        )

    else:
        mask_loss = 0.

    #trainable_variables = list(itertools.chain.from_iterable([model.trainable_variables for model in MODELS.values()]))
    trainable_variables = model.trainable_variables

    if params['optimizer_type'] in ['LAMB', 'Novograd']: # decoupled weight decay
        l2_regularization_loss = tf.constant(0.0)
    else:
        l2_regularization_loss = params['l2_weight_decay'] * tf.add_n([
            tf.nn.l2_loss(v)
            for v in trainable_variables
            if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
        ])

    total_loss = total_rpn_loss + total_fast_rcnn_loss + mask_loss + l2_regularization_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        # Predictions can only contain a dict of tensors, not a dict of dict of
        # tensors. These outputs are not used for eval purposes.
        del predictions['rpn_score_outputs']
        del predictions['rpn_box_outputs']

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=total_loss
        )

    if mode == tf.estimator.ModeKeys.TRAIN:

        if params['lr_schedule'] == 'piecewise':
            learning_rate = learning_rates.step_learning_rate_with_linear_warmup(
                global_step=global_step,
                init_learning_rate=params['init_learning_rate'],
                warmup_learning_rate=params['warmup_learning_rate'],
                warmup_steps=params['warmup_steps'],
                learning_rate_levels=params['learning_rate_levels'],
                learning_rate_steps=params['learning_rate_steps']
            )
        elif params['lr_schedule'] == 'cosine':
            learning_rate = learning_rates.cosine_learning_rate_with_linear_warmup(
                global_step=global_step,
                init_learning_rate=params['init_learning_rate'],
                warmup_learning_rate=params['warmup_learning_rate'],
                warmup_steps=params['warmup_steps'],
                first_decay_steps=params['total_steps'],
                alpha= 0.001
            )
        else:
            raise NotImplementedError

        if params['optimizer_type'] == 'SGD':
            optimizer = create_optimizer(learning_rate, params)
        elif params['optimizer_type'] == 'LAMB':
            optimizer = create_lamb_optimizer(learning_rate, params)
        elif params['optimizer_type'] == 'Novograd':
            optimizer = create_novograd_optimizer(learning_rate, params)
        else:
            raise NotImplementedError

        grads_and_vars = optimizer.compute_gradients(total_loss, trainable_variables, colocate_gradients_with_ops=True)

        gradients, variables = zip(*grads_and_vars)
        
        global_gradient_clip_ratio = params['global_gradient_clip_ratio']
        if global_gradient_clip_ratio > 0.0:
            all_are_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in gradients])
            (clipped_grads, _) = tf.clip_by_global_norm(gradients, clip_norm=global_gradient_clip_ratio,
                                use_norm=tf.cond(all_are_finite, lambda: tf.linalg.global_norm(gradients), lambda: tf.constant(1.0)))
            gradients = clipped_grads
        
        grads_and_vars = []

        # Special treatment for biases (beta is named as bias in reference model)
        # Reference: https://github.com/ddkang/Detectron/blob/80f3295308/lib/modeling/optimizer.py#L109
        for grad, var in zip(gradients, variables):

            if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
                grad = 2.0 * grad

            grads_and_vars.append((grad, var))

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    else:
        train_op = None
        learning_rate = None

    replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group

    if not isinstance(replica_id, tf.Tensor) or tf.get_static_value(replica_id) == 0:

        register_metric(name="L2 loss", tensor=l2_regularization_loss, aggregator=StandardMeter())
        register_metric(name="Mask loss", tensor=mask_loss, aggregator=StandardMeter())
        register_metric(name="Total loss", tensor=total_loss, aggregator=StandardMeter())

        register_metric(name="RPN box loss", tensor=rpn_box_loss, aggregator=StandardMeter())
        register_metric(name="RPN score loss", tensor=rpn_score_loss, aggregator=StandardMeter())
        register_metric(name="RPN total loss", tensor=total_rpn_loss, aggregator=StandardMeter())

        register_metric(name="FastRCNN class loss", tensor=fast_rcnn_class_loss, aggregator=StandardMeter())
        register_metric(name="FastRCNN box loss", tensor=fast_rcnn_box_loss, aggregator=StandardMeter())
        register_metric(name="FastRCNN total loss", tensor=total_fast_rcnn_loss, aggregator=StandardMeter())

        register_metric(name="Learning rate", tensor=learning_rate, aggregator=StandardMeter())
        pass
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
    )


def mask_rcnn_model_fn(features, labels, mode, params):
    """Mask-RCNN model."""

    return _model_fn(
        features,
        labels,
        mode,
        params
    )

class SessionModel(object):
    
    def __init__(self, run_config, train_input_fn=None, eval_input_fn=None, 
                 is_training=True, **kwargs):
        self.run_config = run_config
        self.forward = MRCNN(run_config.values(), is_training=True, **kwargs)
        train_params = dict(run_config.values(), batch_size=run_config.train_batch_size)
        self.train_tdf = tf.compat.v1.data.make_initializable_iterator(train_input_fn(train_params)) \
                            if train_input_fn else None
        eval_params = dict(run_config.values(), batch_size=run_config.eval_batch_size)
        self.eval_tdf = tf.compat.v1.data.make_initializable_iterator(eval_input_fn(eval_params)) \
                            if eval_input_fn else None
        self.train_step = self.train_fn()
        self.eval_step = self.eval_fn()
        
    def train_fn(self):
        global_step = tf.compat.v1.train.get_or_create_global_step()
        features, labels = self.train_tdf.get_next()
        model_outputs = self.forward(features, labels, self.run_config.values(), True)
        model_outputs.update({
            'source_id': features['source_ids'],
            'image_info': features['image_info'],
        })
        # score_loss and box_loss are for logging. only total_loss is optimized.
        total_rpn_loss, rpn_score_loss, rpn_box_loss = losses.rpn_loss(
            score_outputs=model_outputs['rpn_score_outputs'],
            box_outputs=model_outputs['rpn_box_outputs'],
            labels=labels,
            params=self.run_config.values()
        )
        total_fast_rcnn_loss, fast_rcnn_class_loss, fast_rcnn_box_loss = losses.fast_rcnn_loss(
            class_outputs=model_outputs['class_outputs'],
            box_outputs=model_outputs['box_outputs'],
            class_targets=model_outputs['class_targets'],
            box_targets=model_outputs['box_targets'],
            rpn_box_rois=model_outputs['box_rois'],
            image_info=features['image_info'],
            params=self.run_config.values()
        )
        if self.run_config.include_mask:
            mask_loss = losses.mask_rcnn_loss(
                mask_outputs=model_outputs['mask_outputs'],
                mask_targets=model_outputs['mask_targets'],
                select_class_targets=model_outputs['selected_class_targets'],
                params=self.run_config.values()
            )

        else:
            mask_loss = 0.
        trainable_variables = self.forward.trainable_variables
        if self.run_config.optimizer_type in ['LAMB', 'Novograd']: # decoupled weight decay
            l2_regularization_loss = tf.constant(0.0)
        else:
            l2_regularization_loss = self.run_config.l2_weight_decay * tf.add_n([
                tf.nn.l2_loss(v)
                for v in trainable_variables
                if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
            ])
        total_loss = total_rpn_loss + total_fast_rcnn_loss + mask_loss + l2_regularization_loss
        
        if self.run_config.lr_schedule == 'piecewise':
            learning_rate = learning_rates.step_learning_rate_with_linear_warmup(
                global_step=global_step,
                init_learning_rate=self.run_config.init_learning_rate,
                warmup_learning_rate=self.run_config.warmup_learning_rate,
                warmup_steps=self.run_config.warmup_steps,
                learning_rate_levels=self.run_config.learning_rate_levels,
                learning_rate_steps=self.run_config.learning_rate_steps
            )
        elif self.run_config.lr_schedule == 'cosine':
            learning_rate = learning_rates.cosine_learning_rate_with_linear_warmup(
                global_step=global_step,
                init_learning_rate=self.run_config.init_learning_rate,
                warmup_learning_rate=self.run_config.warmup_learning_rate,
                warmup_steps=self.run_config.warmup_steps,
                first_decay_steps=self.run_config.total_steps,
                alpha= 0.001 #* params['init_learning_rate']
            )
        else:
            raise NotImplementedError

        if self.run_config.optimizer_type == 'SGD':
            optimizer = create_optimizer(learning_rate, self.run_config.values())
        elif self.run_config.optimizer_type == 'LAMB':
            optimizer = create_lamb_optimizer(learning_rate, self.run_config.values())
        elif self.run_config.optimizer_type == 'Novograd':
            optimizer = create_novograd_optimizer(learning_rate, self.run_config.values())
        else:
            raise NotImplementedError

        grads_and_vars = optimizer.compute_gradients(total_loss, trainable_variables, colocate_gradients_with_ops=True)
 
        gradients, variables = zip(*grads_and_vars)
 
        global_gradient_clip_ratio = self.run_config.global_gradient_clip_ratio
        if global_gradient_clip_ratio > 0.0:
            all_are_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in gradients])
            (clipped_grads, _) = tf.clip_by_global_norm(gradients, clip_norm=global_gradient_clip_ratio,
                                use_norm=tf.cond(all_are_finite, lambda: tf.linalg.global_norm(gradients), lambda: tf.constant(1.0)))
            gradients = clipped_grads
        
        grads_and_vars = []

        # Special treatment for biases (beta is named as bias in reference model)
        # Reference: https://github.com/ddkang/Detectron/blob/80f3295308/lib/modeling/optimizer.py#L109
        for grad, var in zip(gradients, variables):

            if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
                grad = 2.0 * grad

            grads_and_vars.append((grad, var))

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        output_dict = {'train_op': train_op,
                  'total_loss': total_loss,
                  'total_rpn_loss': total_rpn_loss,
                  'rpn_score_loss': rpn_score_loss,
                  'rpn_box_loss': rpn_box_loss,
                  'total_fast_rcnn_loss': total_fast_rcnn_loss,
                  'fast_rcnn_class_loss': fast_rcnn_class_loss,
                  'fast_rcnn_box_loss': fast_rcnn_box_loss, 
                  'mask_loss': mask_loss,
                  'l2_regularization_loss': l2_regularization_loss,
                  'learning_rate': learning_rate}
        replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group

        if not isinstance(replica_id, tf.Tensor) or tf.get_static_value(replica_id) == 0:

            register_metric(name="L2 loss", tensor=l2_regularization_loss, aggregator=StandardMeter())
            register_metric(name="Mask loss", tensor=mask_loss, aggregator=StandardMeter())
            register_metric(name="Total loss", tensor=total_loss, aggregator=StandardMeter())

            register_metric(name="RPN box loss", tensor=rpn_box_loss, aggregator=StandardMeter())
            register_metric(name="RPN score loss", tensor=rpn_score_loss, aggregator=StandardMeter())
            register_metric(name="RPN total loss", tensor=total_rpn_loss, aggregator=StandardMeter())

            register_metric(name="FastRCNN class loss", tensor=fast_rcnn_class_loss, aggregator=StandardMeter())
            register_metric(name="FastRCNN box loss", tensor=fast_rcnn_box_loss, aggregator=StandardMeter())
            register_metric(name="FastRCNN total loss", tensor=total_fast_rcnn_loss, aggregator=StandardMeter())

            register_metric(name="Learning rate", tensor=learning_rate, aggregator=StandardMeter())
            pass
        return output_dict
 
    def eval_fn(self):
        #with tf.xla.experimental.jit_scope(compile_ops=False):
        features = self.eval_tdf.get_next()['features']
        labels = None
        model_outputs = self.forward(features, labels, self.run_config.values(), False)
        model_outputs.update({
                'source_id': features['source_ids'],
                'image_info': features['image_info'],
            })
        return model_outputs
    
    @staticmethod
    def get_session_config(use_xla=True, use_amp=True, 
                            use_tf_distributed=False, allow_xla_at_inference=False):
        rewrite_options = rewriter_config_pb2.RewriterConfig(
            meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.TWO)
        if use_amp:
            rewrite_options.auto_mixed_precision = True
        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            graph_options=tf.compat.v1.GraphOptions(
                rewrite_options=rewrite_options,
                # infer_shapes=True  # Heavily drops throughput by 30%
            )
        )
        if use_tf_distributed:
            config.gpu_options.force_gpu_compatible = False
        else:
            config.gpu_options.force_gpu_compatible = True
            if MPI_is_distributed():
                config.gpu_options.visible_device_list = str(MPI_local_rank())
        if use_xla:
            logging.info("XLA is activated - Experiment Feature")
            config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
        config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads
        if MPI_is_distributed():
            config.inter_op_parallelism_threads = max(2, mp.cpu_count() // hvd.local_size())
        elif not use_tf_distributed:
            config.inter_op_parallelism_threads = 4
        return config
    
class TapeModel(object):
    
    def __init__(self, params, train_input_fn=None, eval_input_fn=None, is_training=True):
        self.params = params


        self.forward = MRCNN(self.params.values(), is_training=is_training)
        self.model_dir = self.params.model_dir
        train_params = dict(self.params.values(), batch_size=self.params.train_batch_size)
        self.train_tdf = iter(train_input_fn(train_params)) \
                            if train_input_fn else None
        eval_params = dict(self.params.values(), batch_size=self.params.eval_batch_size)
        self.eval_tdf = iter(eval_input_fn(eval_params)) \
                            if eval_input_fn else None
        self.optimizer, self.schedule = self.get_optimizer()
        self.epoch_num = 0

    def load_weights(self):
        chkp = tf.compat.v1.train.NewCheckpointReader(self.params.checkpoint)
        weights = [chkp.get_tensor(i) for i in eager_mapping.resnet_vars]
        self.forward.layers[0].set_weights(weights)
        
    def get_optimizer(self):
        if self.params.lr_schedule=='piecewise':
            schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.params.learning_rate_steps,
                                                                            [self.params.init_learning_rate] + \
                                                                            self.params.learning_rate_levels)
        elif self.params.lr_schedule=='cosine':
            schedule = tf.keras.experimental.CosineDecay(self.params.init_learning_rate,
                                                         self.params.total_steps,
                                                         alpha=0.001)
        else:
            raise NotImplementedError
        schedule = warmup_scheduler.WarmupScheduler(schedule, self.params.warmup_learning_rate,
                                                    self.params.warmup_steps)
        if self.params.optimizer_type=="SGD":
            opt = tf.keras.optimizers.SGD(learning_rate=schedule, 
                                          momentum=self.params.momentum)
        elif self.params.optimizer_type=="LAMB":
            opt = tfa.optimizers.LAMB(learning_rate=schedule)
        elif self.params.optimizer_type=="Novograd":
            opt = optimizers.NovoGrad(learning_rate=schedule,
                                          beta_1=self.params.beta1,
                                          beta_2=self.params.beta2,
                                          weight_decay=self.params.l2_weight_decay,
                                          exclude_from_weight_decay=['bias', 'beta', 'batch_normalization'])
        else:
            raise NotImplementedError
        if self.params.amp:
            opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
        return opt, schedule


    @tf.function
    def train_step(self, features, labels, sync_weights=False, sync_opt=False):
        loss_dict = dict()
        with tf.GradientTape() as tape:
            model_outputs = self.forward(features, labels, self.params.values(), True)
            loss_dict['total_rpn_loss'], loss_dict['rpn_score_loss'], \
                loss_dict['rpn_box_loss'] = losses.rpn_loss(
                    score_outputs=model_outputs['rpn_score_outputs'],
                    box_outputs=model_outputs['rpn_box_outputs'],
                    labels=labels,
                    params=self.params.values()
                )
            loss_dict['total_fast_rcnn_loss'], loss_dict['fast_rcnn_class_loss'], \
                loss_dict['fast_rcnn_box_loss'] = losses.fast_rcnn_loss(
                    class_outputs=model_outputs['class_outputs'],
                    box_outputs=model_outputs['box_outputs'],
                    class_targets=model_outputs['class_targets'],
                    box_targets=model_outputs['box_targets'],
                    rpn_box_rois=model_outputs['box_rois'],
                    image_info=features['image_info'],
                    params=self.params.values()
                )
            if self.params.include_mask:
                loss_dict['mask_loss'] = losses.mask_rcnn_loss(
                    mask_outputs=model_outputs['mask_outputs'],
                    mask_targets=model_outputs['mask_targets'],
                    select_class_targets=model_outputs['selected_class_targets'],
                    params=self.params.values()
                )
            else:
                loss_dict['mask_loss'] = 0.
            if self.params.optimizer_type in ['LAMB', 'Novograd']: # decoupled weight decay
                loss_dict['l2_regularization_loss'] = tf.constant(0.0)
            else:
                loss_dict['l2_regularization_loss'] = self.params.l2_weight_decay * tf.add_n([
                    tf.nn.l2_loss(v)
                    for v in self.forward.trainable_variables
                    if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
                ])
            loss_dict['total_loss'] = loss_dict['total_rpn_loss'] \
                + loss_dict['total_fast_rcnn_loss'] + loss_dict['mask_loss'] \
                + loss_dict['l2_regularization_loss']
            if self.params.amp:
                scaled_loss = self.optimizer.get_scaled_loss(loss_dict['total_loss'])

        if is_herring():
            if MPI_is_distributed(True):
                tape = herring.DistributedGradientTape(tape)
            if self.params.amp:
                scaled_gradients = tape.gradient(scaled_loss, self.forward.trainable_variables)
                gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = tape.gradient(loss_dict['total_loss'], self.forward.trainable_variables)
            global_gradient_clip_ratio = self.params.global_gradient_clip_ratio
            if global_gradient_clip_ratio > 0.0:
                all_are_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in gradients])
                (clipped_grads, _) = tf.clip_by_global_norm(gradients, clip_norm=global_gradient_clip_ratio,
                                use_norm=tf.cond(all_are_finite, lambda: tf.linalg.global_norm(gradients), lambda: tf.constant(1.0)))
                gradients = clipped_grads

            grads_and_vars = []
            # Special treatment for biases (beta is named as bias in reference model)
            # Reference: https://github.com/ddkang/Detectron/blob/80f3295308/lib/modeling/optimizer.py#L109
            for grad, var in zip(gradients, self.forward.trainable_variables):
                if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
                    grad = 2.0 * grad
                grads_and_vars.append((grad, var))

            # self.optimizer.apply_gradients(zip(gradients, self.forward.trainable_variables))
            self.optimizer.apply_gradients(grads_and_vars) 
            if MPI_is_distributed(True) and sync_weights:
                if MPI_rank(True)==0:
                    logging.info("Broadcasting variables")
                herring.broadcast_variables(self.forward.variables, 0)
            if MPI_is_distributed(True) and sync_opt:
                if MPI_rank(True)==0:
                    logging.info("Broadcasting optimizer")
                herring.broadcast_variables(self.optimizer.variables(), 0)        
        else:
            if MPI_is_distributed():
                tape = hvd.DistributedGradientTape(tape, compression=hvd.compression.NoneCompressor)
            if self.params.amp:
                scaled_gradients = tape.gradient(scaled_loss, self.forward.trainable_variables)
                gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = tape.gradient(loss_dict['total_loss'], self.forward.trainable_variables)
            global_gradient_clip_ratio = self.params.global_gradient_clip_ratio
            if global_gradient_clip_ratio > 0.0:
                all_are_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in gradients])
                (clipped_grads, _) = tf.clip_by_global_norm(gradients, clip_norm=global_gradient_clip_ratio,
                                use_norm=tf.cond(all_are_finite, lambda: tf.linalg.global_norm(gradients), lambda: tf.constant(1.0)))
                gradients = clipped_grads
        
            grads_and_vars = []
            # Special treatment for biases (beta is named as bias in reference model)
            # Reference: https://github.com/ddkang/Detectron/blob/80f3295308/lib/modeling/optimizer.py#L109
            for grad, var in zip(gradients, self.forward.trainable_variables):
                if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
                    grad = 2.0 * grad
                grads_and_vars.append((grad, var))

            # self.optimizer.apply_gradients(zip(gradients, self.forward.trainable_variables))
            self.optimizer.apply_gradients(grads_and_vars)

            if MPI_is_distributed() and sync_weights:
                if MPI_rank()==0:
                    logging.info("Broadcasting variables")
                hvd.broadcast_variables(self.forward.variables, 0)
            if MPI_is_distributed() and sync_opt:
                if MPI_rank()==0:
                    logging.info("Broadcasting optimizer")
                hvd.broadcast_variables(self.optimizer.variables(), 0)
        return loss_dict
    
    def initialize_model(self):
        features, labels = next(self.train_tdf)
        model_outputs = self.forward(features, labels, self.params.values(), True)
        self.load_weights()
    
    def initialize_eval_model(self, features):
        for _ in range(5):
          _ = self.predict(features)
          
    def train_epoch(self, steps, broadcast=False, profile=None):
        if MPI_rank(is_herring())==0:
            logging.info("Starting training loop")
            p_bar = tqdm(range(steps), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            loss_history = []
        else:
            p_bar = range(steps)

        times=[]

        if MPI_rank(is_herring())==0 and profile is not None:
            logging.info(f"Saving profile to {profile}")
            tf_profiler.start(profile)
            for i in p_bar:
                if broadcast and i==0:
                    b_w, b_o = True, True
                elif i==0:
                    b_w, b_o = False, True
                else:
                    b_w, b_o = False, False

                b_w = tf.convert_to_tensor(b_w)
                b_o = tf.convert_to_tensor(b_o)
                with prof_Trace(f"Step-",step_num=i,_r=1):
                    tstart=time.perf_counter()
                    features, labels = next(self.train_tdf)
                    loss_dict = self.train_step(features, labels, b_w, b_o)
                    times.append(time.perf_counter()-tstart)
                if MPI_rank(is_herring())==0:
                    loss_history.append(loss_dict['total_loss'].numpy())
                    step = self.optimizer.iterations
                    learning_rate = self.schedule(step)
                    p_bar.set_description("Loss: {0:.4f}, LR: {1:.4f}".format(mean(loss_history[-50:]), 
                                                                            learning_rate))
            tf_profiler.stop()
        else:
            runtype="TrainStep"
            for i in p_bar:
                if broadcast and i==0:
                    b_w, b_o = True, True
                elif i==0:
                    b_w, b_o = False, True
                else:
                    b_w, b_o = False, False
                tstart=time.perf_counter()
                step_token=nvtx.push(f"{runtype}-{i}",runtype)
                features, labels = next(self.train_tdf)
                b_w = tf.convert_to_tensor(b_w)
                b_o = tf.convert_to_tensor(b_o)
                loss_dict = self.train_step(features, labels, b_w, b_o)
                times.append(time.perf_counter()-tstart)
                nvtx.pop(step_token)
                if MPI_rank(is_herring())==0:
                    loss_history.append(loss_dict['total_loss'].numpy())
                    step = self.optimizer.iterations
                    learning_rate = self.schedule(step)
                    p_bar.set_description("Loss: {0:.4f}, LR: {1:.4f}".format(mean(loss_history[-50:]), 
                                                                            learning_rate))

        logging.info(f"Rank={MPI_rank()} Avg step time {np.mean(times[1:])*1000.} +/- {np.std(times[1:])*1000.} ms")            

    def get_latest_checkpoint(self):
        try:
            return sorted([_ for _ in os.listdir(self.model_dir) if _.endswith(".h5")])[-1]
        except:
            return None

    def save_model(self):
        filename = os.path.join(self.model_dir, f'weights_{self.epoch_num:02d}.h5')
        f = h5py.File(filename,'w')
        weights = self.forward.get_weights()
        for i in range(len(weights)):
            f.create_dataset('weight'+str(i),data=weights[i])
        f.close()

    def load_model(self, filename):
        file=h5py.File(filename,'r')
        weights = []
        for i in range(len(file.keys())):
            weights.append(file['weight'+str(i)][:])
        self.forward.set_weights(weights)
    
    @tf.function            
    def predict(self, features):
        labels = None
        model_outputs = self.forward(features, labels, self.params.values(), is_training=False)
        model_outputs.update({
                'source_id': features['source_ids'],
                'image_info': features['image_info'],
            })
        return model_outputs
    #@profile_dec
    def run_eval(self, steps, batches, async_eval=False, use_ext=False, use_dist_coco_eval=False):
        #steps = 5
        if MPI_rank(is_herring())==0:
            logging.info("Starting eval loop")
            p_bar = tqdm(range(steps), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        else:
            p_bar = range(steps)
        worker_predictions = dict()
        data_load_total = 0
        predict_total = 0
        process_total = 0
        append_total = 0
        MPI.COMM_WORLD.barrier()
        start_total_infer = time.time()
        
        in_q = mp.Queue()
        out_q = mp.Queue()
        stop_event = mp.Event()
        stop_event.clear()
        post_proc = mp.Process(target=coco_pre_process, args=(in_q, out_q, stop_event))
        post_proc2 = mp.Process(target=coco_pre_process, args=(in_q, out_q, stop_event))
        post_proc3 = mp.Process(target=coco_pre_process, args=(in_q, out_q, stop_event))
        post_proc.start()
        post_proc2.start()
        post_proc3.start()

        #if MPI_rank(is_herring())==0:
        #  tf.profiler.experimental.start('logdir')

        for i in p_bar:
            start = time.time()
            features = batches[i]#next(self.eval_tdf)['features']
            data_load_time = time.time()
            data_load_total += data_load_time-start
            
            out = self.predict(features)
            predict_time = time.time()
            predict_total += predict_time - data_load_time
            #Extract numpy from tensors
            for key in out:
              out[key] = out[key].numpy()
            in_q.put(out)
        while(in_q.qsize() > 0):
          time.sleep(.5)
        stop_event.set()
        while(out_q.qsize() < 3):
          time.sleep(.5)
        #Should expect num threads items in queue
        converted_predictions = out_q.get() + out_q.get() + out_q.get()
        post_proc.join()
        post_proc2.join()
        post_proc3.join()
        print("Q is empty ", in_q.empty())
        #if MPI_rank(is_herring())==0:
        #  tf.profiler.experimental.stop()

        
        end_total_infer = time.time()
        MPI.COMM_WORLD.barrier()
        if(not use_dist_coco_eval):
          print(len(converted_predictions), flush=True)
          predictions_list = evaluation.gather_result_from_all_processes(converted_predictions)
          
          validation_json_file=self.params.val_json_file
          end_gather_result = time.time()
          #with cProfile.Profile() as pr:
          if MPI_rank(is_herring()) == 0:
              all_predictions = predictions_list
              print(len(all_predictions), flush=True)
              if use_ext:
                  args = [all_predictions, validation_json_file, use_ext, False]
                  if async_eval:
                      eval_thread = threading.Thread(target=evaluation.fast_eval,
                                                    name="eval-thread", args=args)
                      eval_thread.start()
                  else:
                      evaluation.fast_eval(*args)
              else:
                  args = [all_predictions, source_ids, True, validation_json_file]
                  if async_eval:
                      eval_thread = threading.Thread(target=evaluation.compute_coco_eval_metric_n, 
                                                    name="eval-thread", args=args)
                      eval_thread.start()
                  else:
                      evaluation.compute_coco_eval_metric_n(*args)
        else:
          end_gather_result = time.time()
          validation_json_file=self.params.val_json_file
          evaluation.fast_eval(converted_predictions, validation_json_file, use_ext, use_dist_coco_eval)

        end_coco_eval = time.time()
        if(MPI_rank(is_herring()) == 0):
          print(f"(avg, total) DataLoad ({data_load_total/steps}, {data_load_total}) predict ({predict_total/steps}, {predict_total})")
          print(f"Total Time {end_coco_eval-start_total_infer} Total Infer {end_total_infer - start_total_infer} gather res {end_gather_result - end_total_infer} coco_eval {end_coco_eval - end_gather_result}")

#@profile_dec
def coco_pre_process(in_q, out_q, finish_input):
      
      coco = coco_metric.MaskCOCO()
      #wait until event is set
      converted_predictions = []
      total_preproc = 0
      preproc_cnt = 0
      total_batches_processed = 0
      while(not finish_input.is_set() or in_q.qsize() > 0):
        try:
          start = time.time()
          out = in_q.get(timeout=0.5)
          start_q = time.time()
          preproc_cnt +=1
          worker_predictions = {}
          total_batches_processed += len(out['detection_scores'])
          out = evaluation.process_prediction_for_eval_batch(out)
          for k, v in out.items():
              if k not in worker_predictions:
                  worker_predictions[k] = [v]
              else:
                  worker_predictions[k].append(v)
          for k, v in worker_predictions.items():
              worker_predictions[k] = np.concatenate(v, axis=0)
          #print(len(worker_predictions), flush=True)
          # score_threshold = .2
          # print(out['detection_scores'].shape, flush=True)
          
          # for ii in range(len(out['detection_scores'])):
          #   thold = out['detection_scores'][ii] > score_threshold
          #   thold_shape = out['detection_scores'][ii].shape
          #   for key in out:
          #     print(out[key][ii].shape, thold_shape)
          #     if(out[key][ii].shape != thold_shape):
          #       print(key)
          #       continue
          #     out[key][ii] = out[key][ii][thold]
          
          # print("POST",out['detection_scores'].shape, flush=True)

          converted_predictions += coco.load_predictions(worker_predictions, include_mask=True, is_image_mask=False)
          end_coco_load = time.time()
          total_preproc += end_coco_load - start
        except queue.Empty:
          pass
      print(not in_q.empty(), "Converted preds in mp ",len(converted_predictions), " ", total_batches_processed, flush=True)
      out_q.put(converted_predictions)
      #print(f"Time taken to process outputs {total_preproc/preproc_cnt}/{total_preproc}")
      return