import os
import sys
import itertools
import tensorflow.compat.v1 as tf
import horovod.tensorflow as hvd

from mask_rcnn import anchors
from mask_rcnn.ops import roi_ops
from mask_rcnn.models import resnet
from mask_rcnn.models import fpn
from mask_rcnn.models import heads
from mask_rcnn.utils import box_utils
from mask_rcnn.ops import postprocess_ops
from mask_rcnn.ops import roi_ops
from mask_rcnn.ops import spatial_transform_ops
from mask_rcnn.ops import training_ops
from mask_rcnn.training import losses, learning_rates

MODELS = dict()

def forward(features_0, features_1, params, devices, labels_0=None, labels_1=None, is_training=True):
    model_outputs = {}
    is_gpu_inference = not is_training and params['use_batched_nms']
    batch_size, image_height, image_width, _ = features_0['images'].get_shape().as_list()
    image_width = image_width*2
    with tf.device(devices[0].name):
        all_anchors_0 = anchors.Anchors(params['min_level'], params['max_level'],
                                  params['num_scales'], params['aspect_ratios'],
                                  params['anchor_scale'],
                                  (image_height, image_width))
    with tf.device(devices[1].name):
        all_anchors_1 = anchors.Anchors(params['min_level'], params['max_level'],
                                  params['num_scales'], params['aspect_ratios'],
                                  params['anchor_scale'],
                                  (image_height, image_width))
    
    # Place ops on each device
    with tf.device(devices[0].name):
        MODELS["backbone_0"] = resnet.Resnet_Model(
                            "resnet50",
                            data_format='channels_last',
                            trainable=is_training,
                            finetune_bn=params['finetune_bn']
                        )
    with tf.device(devices[1].name):
        MODELS["backbone_1"] = resnet.Resnet_Model(
                            "resnet50",
                            data_format='channels_last',
                            trainable=is_training,
                            finetune_bn=params['finetune_bn']
                        )
    # Run half of resnet on each
    with tf.device(devices[0].name):
        res_out_0 = MODELS["backbone_0"](features_0['images'])
        res_out_0[2] = res_out_0[2][:,:,:-16,:]
        res_out_0[3] = res_out_0[3][:,:,:-8,:]
        res_out_0[4] = res_out_0[4][:,:,:-4,:]
        res_out_0[5] = res_out_0[5][:,:,:-2,:]
    
    with tf.device(devices[1].name):
        res_out_1 = MODELS["backbone_1"](features_1['images'])
        res_out_1[2] = res_out_1[2][:,:,16:,:]
        res_out_1[3] = res_out_1[3][:,:,8:,:]
        res_out_1[4] = res_out_1[4][:,:,4:,:]
        res_out_1[5] = res_out_1[5][:,:,2:,:]
    
    # copy tensors from device 1 to device 0 cut off overlapping section
    with tf.device(devices[0].name):
        C2_1_0 = tf.stop_gradient(tf.identity(res_out_1[2]))
        C3_1_0 = tf.stop_gradient(tf.identity(res_out_1[3]))
        C4_1_0 = tf.stop_gradient(tf.identity(res_out_1[4]))
        C5_1_0 = tf.stop_gradient(tf.identity(res_out_1[5]))
    
    # copy tensors from device 0 to device 1 cut off overlapping section
    with tf.device(devices[1].name):
        C2_0_1 = tf.stop_gradient(tf.identity(res_out_0[2]))
        C3_0_1 = tf.stop_gradient(tf.identity(res_out_0[3]))
        C4_0_1 = tf.stop_gradient(tf.identity(res_out_0[4]))
        C5_0_1 = tf.stop_gradient(tf.identity(res_out_0[5]))
    
    # concatenate tensors
    with tf.device(devices[0].name):
        res_out_0[2] = tf.concat([res_out_0[2], C2_1_0], axis=2)
        res_out_0[3] = tf.concat([res_out_0[3], C3_1_0], axis=2)
        res_out_0[4] = tf.concat([res_out_0[4], C4_1_0], axis=2)
        res_out_0[5] = tf.concat([res_out_0[5], C5_1_0], axis=2)
    
    with tf.device(devices[1].name):
        res_out_1[2] = tf.concat([C2_0_1, res_out_1[2]], axis=2)
        res_out_1[3] = tf.concat([C3_0_1, res_out_1[3]], axis=2)
        res_out_1[4] = tf.concat([C4_0_1, res_out_1[4]], axis=2)
        res_out_1[5] = tf.concat([C5_0_1, res_out_1[5]], axis=2)
    
    # run full FPN and RPN on both GPUs
    with tf.device(devices[0].name):
        MODELS["FPN_0"] = fpn.FPNNetwork(params['min_level'], 
                                       params['max_level'], 
                                       trainable=is_training)
        fpn_feats_0 = MODELS["FPN_0"](res_out_0, training=is_training)
    
        # model_outputs.update({'fpn_features_0': fpn_feats_0})

        def rpn_head_0_fn(features, min_level=2, max_level=6, num_anchors=3):
            """Region Proposal Network (RPN) for Mask-RCNN."""
            scores_outputs = dict()
            box_outputs = dict()

            MODELS["RPN_Heads_0"] = heads.RPN_Head_Model(name="rpn_head_0", 
                                                       num_anchors=num_anchors, 
                                                       trainable=is_training)

            for level in range(min_level, max_level + 1):
                scores_outputs[level], box_outputs[level] = \
                MODELS["RPN_Heads_0"](features[level], 
                                    training=is_training)
                scores_outputs[level] = tf.cast(scores_outputs[level], tf.float32)
                box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
            return scores_outputs, box_outputs

        rpn_score_outputs_0, rpn_box_outputs_0 = rpn_head_0_fn(
            features=fpn_feats_0,
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

        rpn_box_scores_0, rpn_box_rois_0 = roi_ops.custom_multilevel_propose_rois(
            scores_outputs=rpn_score_outputs_0,
            box_outputs=rpn_box_outputs_0,
            all_anchors=all_anchors_0,
            image_info=features_0['image_info'],
            rpn_pre_nms_topn=rpn_pre_nms_topn,
            rpn_post_nms_topn=rpn_post_nms_topn,
            rpn_nms_threshold=rpn_nms_threshold,
            rpn_min_size=params['rpn_min_size']
        )
        
    with tf.device(devices[1].name):
        MODELS["FPN_1"] = fpn.FPNNetwork(params['min_level'], 
                                       params['max_level'], 
                                       trainable=is_training)
        fpn_feats_1 = MODELS["FPN_1"](res_out_1, training=is_training)
    
        # model_outputs.update({'fpn_features_0': fpn_feats_0})

        def rpn_head_1_fn(features, min_level=2, max_level=6, num_anchors=3):
            """Region Proposal Network (RPN) for Mask-RCNN."""
            scores_outputs = dict()
            box_outputs = dict()

            MODELS["RPN_Heads_1"] = heads.RPN_Head_Model(name="rpn_head_1", 
                                                       num_anchors=num_anchors, 
                                                       trainable=is_training)

            for level in range(min_level, max_level + 1):
                scores_outputs[level], box_outputs[level] = \
                MODELS["RPN_Heads_1"](features[level], 
                                    training=is_training)
                scores_outputs[level] = tf.cast(scores_outputs[level], tf.float32)
                box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
            return scores_outputs, box_outputs

        rpn_score_outputs_1, rpn_box_outputs_1 = rpn_head_1_fn(
            features=fpn_feats_1,
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

        rpn_box_scores_1, rpn_box_rois_1 = roi_ops.custom_multilevel_propose_rois(
            scores_outputs=rpn_score_outputs_1,
            box_outputs=rpn_box_outputs_1,
            all_anchors=all_anchors_1,
            image_info=features_1['image_info'],
            rpn_pre_nms_topn=rpn_pre_nms_topn,
            rpn_post_nms_topn=rpn_post_nms_topn,
            rpn_nms_threshold=rpn_nms_threshold,
            rpn_min_size=params['rpn_min_size']
        )
    
    # No GPU op for sampling
    if is_training:
        rpn_box_rois_0 = tf.stop_gradient(rpn_box_rois_0)
        rpn_box_scores_0 = tf.stop_gradient(rpn_box_scores_0)
        rpn_box_rois_1 = tf.stop_gradient(rpn_box_rois_1)
        rpn_box_scores_1 = tf.stop_gradient(rpn_box_scores_1)
        # Sampling
        box_targets_0, class_targets_0, rpn_box_rois_0, proposal_to_label_map_0 = \
        training_ops.proposal_label_op(
            rpn_box_rois_0,
            labels_0['gt_boxes'],
            labels_0['gt_classes'],
            batch_size_per_im=params['batch_size_per_im'],
            fg_fraction=params['fg_fraction'],
            fg_thresh=params['fg_thresh'],
            bg_thresh_hi=params['bg_thresh_hi'],
            bg_thresh_lo=params['bg_thresh_lo']
        )
        box_targets_1, class_targets_1, rpn_box_rois_1, proposal_to_label_map_1 = \
        training_ops.proposal_label_op(
            rpn_box_rois_1,
            labels_1['gt_boxes'],
            labels_1['gt_classes'],
            batch_size_per_im=params['batch_size_per_im'],
            fg_fraction=params['fg_fraction'],
            fg_thresh=params['fg_thresh'],
            bg_thresh_hi=params['bg_thresh_hi'],
            bg_thresh_lo=params['bg_thresh_lo']
        )
    
    # Run both heads on first GPU
    with tf.device(devices[0].name):
        # Performs multi-level RoIAlign.
        box_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats_0,
            boxes=rpn_box_rois_0,
            output_size=7,
            is_gpu_inference=is_gpu_inference
        )

        MODELS["Box_Head_0"] = heads.Box_Head_Model(
            num_classes=params['num_classes'],
            mlp_head_dim=params['fast_rcnn_mlp_head_dim'],
            trainable=is_training
        )
        class_outputs_0, box_outputs_0, _ = MODELS["Box_Head_0"](inputs=box_roi_features)
        if not is_training:
            generate_detections_fn = postprocess_ops.generate_detections_gpu
            detections_0 = generate_detections_fn(
                class_outputs=class_outputs_0,
                box_outputs=box_outputs_0,
                anchor_boxes=rpn_box_rois_0,
                image_info=features_0['image_info'],
                pre_nms_num_detections=params['test_rpn_post_nms_topn'],
                post_nms_num_detections=params['test_detections_per_image'],
                nms_threshold=params['test_nms'],
                bbox_reg_weights=params['bbox_reg_weights']
            )
            model_outputs.update({
                'num_detections_0': detections_0[0],
                'detection_boxes_0': detections_0[1],
                'detection_classes_0': detections_0[2],
                'detection_scores_0': detections_0[3],
            })
        else:  # is training
            encoded_box_targets_0 = training_ops.encode_box_targets(
                boxes=rpn_box_rois_0,
                gt_boxes=box_targets_0,
                gt_labels=class_targets_0,
                bbox_reg_weights=params['bbox_reg_weights']
            )

            model_outputs.update({
                'rpn_score_outputs_0': rpn_score_outputs_0,
                'rpn_box_outputs_0': rpn_box_outputs_0,
                'class_outputs_0': tf.cast(class_outputs_0, tf.float32),
                'box_outputs_0': tf.cast(box_outputs_0, tf.float32),
                'class_targets_0': class_targets_0,
                'box_targets_0': encoded_box_targets_0,
                'box_rois_0': rpn_box_rois_0,
            })
        if not is_training:
            selected_box_rois_0 = model_outputs['detection_boxes_0']
            class_indices_0 = model_outputs['detection_classes_0']
            
            # If using GPU for inference, delay the cast until when Gather ops show up
            # since GPU inference supports float point better.
            # TODO(laigd): revisit this when newer versions of GPU libraries is
            # released.
            if not params['use_batched_nms']:
                class_indices_0 = tf.cast(class_indices_0, dtype=tf.int32)
        
        else:
            selected_class_targets_0, selected_box_targets_0, \
            selected_box_rois_0, proposal_to_label_map_0 = \
            training_ops.select_fg_for_masks(
                class_targets=class_targets_0,
                box_targets=box_targets_0,
                boxes=rpn_box_rois_0,
                proposal_to_label_map=proposal_to_label_map_0,
                max_num_fg=int(params['batch_size_per_im'] * params['fg_fraction'])
            )

            class_indices_0 = tf.cast(selected_class_targets_0, dtype=tf.int32)
            
        mask_roi_features_0 = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats_0,
            boxes=selected_box_rois_0,
            output_size=14,
            is_gpu_inference=is_gpu_inference
        )
        
        MODELS["Mask_Head_0"] = heads.Mask_Head_Model(
            class_indices_0,
            num_classes=params['num_classes'],
            mrcnn_resolution=params['mrcnn_resolution'],
            is_gpu_inference=is_gpu_inference,
            trainable=is_training,
            name="mask_head"
        )

        mask_outputs_0 = MODELS["Mask_Head_0"](inputs=mask_roi_features_0)

        if is_training:
            mask_targets_0 = training_ops.get_mask_targets(

                fg_boxes=selected_box_rois_0,
                fg_proposal_to_label_map=proposal_to_label_map_0,
                fg_box_targets=selected_box_targets_0,
                mask_gt_labels=labels_0['cropped_gt_masks'],
                output_size=params['mrcnn_resolution']
            )

            model_outputs.update({
                'mask_outputs_0': tf.cast(mask_outputs_0, tf.float32),
                'mask_targets_0': mask_targets_0,
                'selected_class_targets_0': selected_class_targets_0,
            })

        else:
            model_outputs.update({
                'detection_masks_0': tf.nn.sigmoid(mask_outputs_0),
            })
            
    # Run both heads on second GPU
    with tf.device(devices[1].name):
        # Performs multi-level RoIAlign.
        box_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats_1,
            boxes=rpn_box_rois_1,
            output_size=7,
            is_gpu_inference=is_gpu_inference
        )

        MODELS["Box_Head_1"] = heads.Box_Head_Model(
            num_classes=params['num_classes'],
            mlp_head_dim=params['fast_rcnn_mlp_head_dim'],
            trainable=is_training
        )
        class_outputs_1, box_outputs_1, _ = MODELS["Box_Head_1"](inputs=box_roi_features)
        if not is_training:
            generate_detections_fn = postprocess_ops.generate_detections_gpu
            detections_1 = generate_detections_fn(
                class_outputs=class_outputs_1,
                box_outputs=box_outputs_1,
                anchor_boxes=rpn_box_rois_1,
                image_info=features_1['image_info'],
                pre_nms_num_detections=params['test_rpn_post_nms_topn'],
                post_nms_num_detections=params['test_detections_per_image'],
                nms_threshold=params['test_nms'],
                bbox_reg_weights=params['bbox_reg_weights']
            )
            model_outputs.update({
                'num_detections_1': detections_1[0],
                'detection_boxes_1': detections_1[1],
                'detection_classes_1': detections_1[2],
                'detection_scores_1': detections_1[3],
            })
        else:  # is training
            encoded_box_targets_1 = training_ops.encode_box_targets(
                boxes=rpn_box_rois_1,
                gt_boxes=box_targets_1,
                gt_labels=class_targets_1,
                bbox_reg_weights=params['bbox_reg_weights']
            )

            model_outputs.update({
                'rpn_score_outputs_1': rpn_score_outputs_1,
                'rpn_box_outputs_1': rpn_box_outputs_1,
                'class_outputs_1': tf.cast(class_outputs_1, tf.float32),
                'box_outputs_1': tf.cast(box_outputs_1, tf.float32),
                'class_targets_1': class_targets_1,
                'box_targets_1': encoded_box_targets_1,
                'box_rois_1': rpn_box_rois_1,
            })
        if not is_training:
            selected_box_rois_1 = model_outputs['detection_boxes_1']
            class_indices_1 = model_outputs['detection_classes_1']
            
            # If using GPU for inference, delay the cast until when Gather ops show up
            # since GPU inference supports float point better.
            # TODO(laigd): revisit this when newer versions of GPU libraries is
            # released.
            if not params['use_batched_nms']:
                class_indices_1 = tf.cast(class_indices_1, dtype=tf.int32)
        
        else:
            selected_class_targets_1, selected_box_targets_1, \
            selected_box_rois_1, proposal_to_label_map_1 = \
            training_ops.select_fg_for_masks(
                class_targets=class_targets_1,
                box_targets=box_targets_1,
                boxes=rpn_box_rois_1,
                proposal_to_label_map=proposal_to_label_map_1,
                max_num_fg=int(params['batch_size_per_im'] * params['fg_fraction'])
            )

            class_indices_1 = tf.cast(selected_class_targets_1, dtype=tf.int32)
            
        mask_roi_features_1 = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats_1,
            boxes=selected_box_rois_1,
            output_size=14,
            is_gpu_inference=is_gpu_inference
        )
        
        MODELS["Mask_Head_1"] = heads.Mask_Head_Model(
            class_indices_1,
            num_classes=params['num_classes'],
            mrcnn_resolution=params['mrcnn_resolution'],
            is_gpu_inference=is_gpu_inference,
            trainable=is_training,
            name="mask_head"
        )

        mask_outputs_1 = MODELS["Mask_Head_1"](inputs=mask_roi_features_1)

        if is_training:
            mask_targets_1 = training_ops.get_mask_targets(

                fg_boxes=selected_box_rois_1,
                fg_proposal_to_label_map=proposal_to_label_map_1,
                fg_box_targets=selected_box_targets_1,
                mask_gt_labels=labels_1['cropped_gt_masks'],
                output_size=params['mrcnn_resolution']
            )

            model_outputs.update({
                'mask_outputs_1': tf.cast(mask_outputs_1, tf.float32),
                'mask_targets_1': mask_targets_1,
                'selected_class_targets_1': selected_class_targets_1,
            })

        else:
            model_outputs.update({
                'detection_masks_1': tf.nn.sigmoid(mask_outputs_1),
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

def model(features_0, features_1, params, devices, labels_0=None, labels_1=None, is_training=True):
    global_step = tf.train.get_or_create_global_step()
    model_outputs = forward(features_0, features_1, params, 
                            devices, labels_0, labels_1, is_training)
    #model_outputs['class_targets'] = tf.cast(model_outputs['class_targets'])
    model_outputs.update({
        'source_id': features_0['source_ids'],
        'image_info': features_0['image_info'],
    })
    if not is_training:
        predictions = {}
        try:
            model_outputs['orig_images'] = features_0['orig_images']
        except KeyError:
            pass
        model_outputs.pop('fpn_features', None)
        predictions.update(model_outputs)
        return model_outputs
    
    trainable_variables = list(itertools.chain.from_iterable([model.trainable_variables \
                                                              for model in MODELS.values()]))
    
    learning_rate = learning_rates.step_learning_rate_with_linear_warmup(
        global_step=global_step,
        init_learning_rate=params['init_learning_rate'],
        warmup_learning_rate=params['warmup_learning_rate'],
        warmup_steps=params['warmup_steps'],
        learning_rate_levels=params['learning_rate_levels'],
        learning_rate_steps=params['learning_rate_steps']
    )
    
    gpu_0_vars = []
    gpu_1_vars = []
    for var in trainable_variables:
        if var.device == '/device:GPU:0':
            gpu_0_vars.append(var)
        elif var.device == '/device:GPU:1':
            gpu_1_vars.append(var)
    
    with tf.device(devices[0].name):
        total_rpn_loss_0, rpn_score_loss_0, rpn_box_loss_0 = losses.rpn_loss(
            score_outputs=model_outputs['rpn_score_outputs_0'],
            box_outputs=model_outputs['rpn_box_outputs_0'],
            labels=labels_0,
            params=params
        )
        total_fast_rcnn_loss_0, fast_rcnn_class_loss_0, fast_rcnn_box_loss_0 = losses.fast_rcnn_loss(
            class_outputs=model_outputs['class_outputs_0'],
            box_outputs=model_outputs['box_outputs_0'],
            class_targets=model_outputs['class_targets_0'],
            box_targets=model_outputs['box_targets_0'],
            params=params
        )
        mask_loss_0 = losses.mask_rcnn_loss(
            mask_outputs=model_outputs['mask_outputs_0'],
            mask_targets=model_outputs['mask_targets_0'],
            select_class_targets=model_outputs['selected_class_targets_0'],
            params=params
        )
        
        l2_regularization_loss_0 = params['l2_weight_decay'] * tf.add_n([
            tf.nn.l2_loss(v)
            for v in gpu_0_vars
            if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
        ])

        total_loss_0 = total_rpn_loss_0 + total_fast_rcnn_loss_0 + mask_loss_0 + l2_regularization_loss_0
        
        optimizer_0 = create_optimizer(learning_rate, params)
        grads_and_vars_0 = optimizer_0.compute_gradients(total_loss_0, 
                                                  gpu_0_vars, 
                                                  colocate_gradients_with_ops=True)
        gradients_0, variables_0 = zip(*grads_and_vars_0)
        grads_and_vars_0 = []
        for grad, var in zip(gradients_0, variables_0):

            if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
                grad = 2.0 * grad

            grads_and_vars_0.append((grad, var))
        train_op_0 = optimizer_0.apply_gradients(grads_and_vars_0, global_step=global_step)
        
    
    with tf.device(devices[1].name):
        total_rpn_loss_1, rpn_score_loss_1, rpn_box_loss_1 = losses.rpn_loss(
            score_outputs=model_outputs['rpn_score_outputs_1'],
            box_outputs=model_outputs['rpn_box_outputs_1'],
            labels=labels_1,
            params=params
        )
        total_fast_rcnn_loss_1, fast_rcnn_class_loss_1, fast_rcnn_box_loss_1 = losses.fast_rcnn_loss(
            class_outputs=model_outputs['class_outputs_1'],
            box_outputs=model_outputs['box_outputs_1'],
            class_targets=model_outputs['class_targets_1'],
            box_targets=model_outputs['box_targets_1'],
            params=params
        )
        mask_loss_1 = losses.mask_rcnn_loss(
            mask_outputs=model_outputs['mask_outputs_1'],
            mask_targets=model_outputs['mask_targets_1'],
            select_class_targets=model_outputs['selected_class_targets_1'],
            params=params
        )
        
        l2_regularization_loss_1 = params['l2_weight_decay'] * tf.add_n([
            tf.nn.l2_loss(v)
            for v in gpu_1_vars
            if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
        ])

        total_loss_1 = total_rpn_loss_1 + total_fast_rcnn_loss_1 + mask_loss_1 + l2_regularization_loss_1
        
        optimizer_1 = create_optimizer(learning_rate, params)
        grads_and_vars_1 = optimizer_1.compute_gradients(total_loss_1, 
                                                  gpu_1_vars, 
                                                  colocate_gradients_with_ops=True)
        gradients_1, variables_1 = zip(*grads_and_vars_1)
        grads_and_vars_1 = []
        for grad, var in zip(gradients_1, variables_1):

            if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
                grad = 2.0 * grad

            grads_and_vars_1.append((grad, var))
        train_op_1 = optimizer_1.apply_gradients(grads_and_vars_1, global_step=global_step)
    
    return train_op_0, train_op_1, total_loss_0, total_loss_1
