import os
import sys
sys.path.append('..')
import itertools

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import horovod.tensorflow as hvd

from mask_rcnn.hyperparameters import dataset_params
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn import dataset_utils

from mask_rcnn import anchors

from mask_rcnn.models import fpn
from mask_rcnn.models import heads
from mask_rcnn.models import resnet

from mask_rcnn.training import losses, learning_rates

from mask_rcnn.ops import postprocess_ops
from mask_rcnn.ops import roi_ops
from mask_rcnn.ops import spatial_transform_ops
from mask_rcnn.ops import training_ops

MODELS = dict()

def forward(features,  params, labels=None, is_training=True):
    """Builds the forward model graph."""
    model_outputs = {}
    is_gpu_inference = not is_training and params['use_batched_nms']

    batch_size, image_height, image_width, _ = features['images'].get_shape().as_list()

    if 'source_ids' not in features:
        features['source_ids'] = -1 * tf.ones([batch_size], dtype=tf.float32)

    all_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                  params['num_scales'], params['aspect_ratios'],
                                  params['anchor_scale'],
                                  (image_height, image_width))

    MODELS["backbone"] = resnet.Resnet_Model(
        "resnet50",
        data_format='channels_last',
        trainable=is_training,
        finetune_bn=params['finetune_bn']
    )

    backbone_feats = MODELS["backbone"](
        features['images'],
        training=is_training,
    )
    MODELS["FPN"] = fpn.FPNNetwork(params['min_level'], params['max_level'], trainable=is_training)
    fpn_feats = MODELS["FPN"](backbone_feats, training=is_training)

    model_outputs.update({'fpn_features': fpn_feats})

    def rpn_head_fn(features, min_level=2, max_level=6, num_anchors=3):
        """Region Proposal Network (RPN) for Mask-RCNN."""
        scores_outputs = dict()
        box_outputs = dict()

        MODELS["RPN_Heads"] = heads.RPN_Head_Model(name="rpn_head", 
                                                   num_anchors=num_anchors, 
                                                   trainable=is_training)

        for level in range(min_level, max_level + 1):
            scores_outputs[level], box_outputs[level] = MODELS["RPN_Heads"](features[level],
                                                                            training=is_training)
            scores_outputs[level] = tf.cast(scores_outputs[level], tf.float32)
            box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
        return scores_outputs, box_outputs

    rpn_score_outputs, rpn_box_outputs = rpn_head_fn(
        features=fpn_feats,
        min_level=params['min_level'],
        max_level=params['max_level'],
        num_anchors=len(params['aspect_ratios'] * params['num_scales'])
    )
    if is_training:
        rpn_pre_nms_topn = params['train_rpn_pre_nms_topn']
        rpn_post_nms_topn = params['train_rpn_post_nms_topn']
        rpn_nms_threshold = params['train_rpn_nms_threshold']

    else:
        rpn_pre_nms_topn = params['test_rpn_pre_nms_topn']
        rpn_post_nms_topn = params['test_rpn_post_nms_topn']
        rpn_nms_threshold = params['test_rpn_nms_thresh']
    
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
    if is_training:
        rpn_box_rois = tf.stop_gradient(rpn_box_rois)
        rpn_box_scores = tf.stop_gradient(rpn_box_scores)  # TODO Jonathan: Unused => Shall keep ?

        # Sampling
        box_targets, class_targets, rpn_box_rois, proposal_to_label_map = training_ops.proposal_label_op(
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

    MODELS["Box_Head"] = heads.Box_Head_Model(
        num_classes=params['num_classes'],
        mlp_head_dim=params['fast_rcnn_mlp_head_dim'],
        trainable=is_training
    )
    class_outputs, box_outputs, _ = MODELS["Box_Head"](inputs=box_roi_features)
    if not is_training:
        generate_detections_fn = postprocess_ops.generate_detections_gpu
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
    else:  # is training
        encoded_box_targets = training_ops.encode_box_targets(
            boxes=rpn_box_rois,
            gt_boxes=box_targets,
            gt_labels=class_targets,
            bbox_reg_weights=params['bbox_reg_weights']
        )

        model_outputs.update({
            'rpn_score_outputs': rpn_score_outputs,
            'rpn_box_outputs': rpn_box_outputs,
            'class_outputs': tf.cast(class_outputs, tf.float32),
            'box_outputs': tf.cast(box_outputs, tf.float32),
            'class_targets': class_targets,
            'box_targets': encoded_box_targets,
            'box_rois': rpn_box_rois,
        })
    # Mask sampling
    if not is_training:
        selected_box_rois = model_outputs['detection_boxes']
        class_indices = model_outputs['detection_classes']

        # If using GPU for inference, delay the cast until when Gather ops show up
        # since GPU inference supports float point better.
        # TODO(laigd): revisit this when newer versions of GPU libraries is
        # released.
        if not params['use_batched_nms']:
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

    MODELS["Mask_Head"] = heads.Mask_Head_Model(
        class_indices,
        num_classes=params['num_classes'],
        mrcnn_resolution=params['mrcnn_resolution'],
        is_gpu_inference=is_gpu_inference,
        trainable=is_training,
        name="mask_head"
    )

    mask_outputs = MODELS["Mask_Head"](inputs=mask_roi_features)
    
    if is_training:
        mask_targets = training_ops.get_mask_targets(

            fg_boxes=selected_box_rois,
            fg_proposal_to_label_map=proposal_to_label_map,
            fg_box_targets=selected_box_targets,
            mask_gt_labels=labels['cropped_gt_masks'],
            output_size=params['mrcnn_resolution']
        )

        model_outputs.update({
            'mask_outputs': tf.cast(mask_outputs, tf.float32),
            'mask_targets': mask_targets,
            'selected_class_targets': selected_class_targets,
        })

    else:
        model_outputs.update({
            'detection_masks': tf.nn.sigmoid(mask_outputs),
        })

    return model_outputs

def create_optimizer(learning_rate, params):
    """Creates optimized based on the specified flags."""

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=params['momentum'])

    '''optimizer = hvd.DistributedOptimizer(
        optimizer,
        name=None,
        device_dense='/gpu:0',
        device_sparse='',
        compression=hvd.Compression.fp16,
        sparse_as_dense=False
    )'''
    optimizer = hvd.DistributedOptimizer(optimizer)
    
    loss_scale = tf.train.experimental.DynamicLossScale(
        initial_loss_scale=(2 ** 12),
        increment_period=2000,
        multiplier=2.0
    )
    optimizer = tf.train.experimental.MixedPrecisionLossScaleOptimizer(optimizer,
                                                                       loss_scale=loss_scale)
    return optimizer

def model(features,  params, labels=None, is_training=True):
    global_step = tf.train.get_or_create_global_step()
    model_outputs = forward(features, params, labels, is_training)
    #model_outputs['class_targets'] = tf.cast(model_outputs['class_targets'])
    model_outputs.update({
        'source_id': features['source_ids'],
        'image_info': features['image_info'],
    })
    if not is_training:
        predictions = {}
        try:
            model_outputs['orig_images'] = features['orig_images']
        except KeyError:
            pass
        model_outputs.pop('fpn_features', None)
        predictions.update(model_outputs)
        return model_outputs
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
        params=params
    )
    mask_loss = losses.mask_rcnn_loss(
        mask_outputs=model_outputs['mask_outputs'],
        mask_targets=model_outputs['mask_targets'],
        select_class_targets=model_outputs['selected_class_targets'],
        params=params
    )
    trainable_variables = list(itertools.chain.from_iterable([model.trainable_variables \
                                                              for model in MODELS.values()]))
    l2_regularization_loss = params['l2_weight_decay'] * tf.add_n([
        tf.nn.l2_loss(v)
        for v in trainable_variables
        if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
    ])

    total_loss = total_rpn_loss + total_fast_rcnn_loss + mask_loss + l2_regularization_loss
    learning_rate = learning_rates.step_learning_rate_with_linear_warmup(
        global_step=global_step,
        init_learning_rate=params['init_learning_rate'],
        warmup_learning_rate=params['warmup_learning_rate'],
        warmup_steps=params['warmup_steps'],
        learning_rate_levels=params['learning_rate_levels'],
        learning_rate_steps=params['learning_rate_steps']
    )
    optimizer = create_optimizer(learning_rate, params)
    grads_and_vars = optimizer.compute_gradients(total_loss, 
                                                 trainable_variables, 
                                                 colocate_gradients_with_ops=True)

    gradients, variables = zip(*grads_and_vars)
    grads_and_vars = []

    # Special treatment for biases (beta is named as bias in reference model)
    # Reference: https://github.com/ddkang/Detectron/blob/80f3295308/lib/modeling/optimizer.py#L109
    for grad, var in zip(gradients, variables):

        if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
            grad = 2.0 * grad

        grads_and_vars.append((grad, var))

    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    return train_op, total_loss
