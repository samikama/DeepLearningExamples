import tensorflow as tf
from mask_rcnn.models import fpn
from mask_rcnn.models import heads
from mask_rcnn.models import resnet
from mask_rcnn import anchors
from mask_rcnn.ops import roi_ops
from mask_rcnn.ops import postprocess_ops
from mask_rcnn.ops import spatial_transform_ops
from mask_rcnn.ops import training_ops

class MaskRCNN(tf.keras.Model):
    
    def __init__(self, params, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.backbone = resnet.Resnet_Model("resnet50",
                                 data_format='channels_last',
                                 trainable=trainable,
                                 finetune_bn=params['finetune_bn']
                                 )
        self.fpn = fpn.FPNNetwork(params['min_level'], params['max_level'], trainable=trainable)
        num_anchors=len(params['aspect_ratios'] * params['num_scales'])
        self.rpn = heads.RPN_Head_Model(name="rpn_head", num_anchors=num_anchors, trainable=trainable)
        is_gpu_inference = not trainable and params['use_batched_nms']
        self.box_head = heads.Box_Head_Model(num_classes=params['num_classes'],
                                             mlp_head_dim=params['fast_rcnn_mlp_head_dim'],
                                             trainable=trainable
                                            )
        if params['include_mask']:
            self.mask_head = heads.Mask_Head_Model(num_classes=params['num_classes'],
                                                   mrcnn_resolution=params['mrcnn_resolution'],
                                                   is_gpu_inference=is_gpu_inference,
                                                   trainable=trainable,
                                                   name="mask_head"
                                                   )
        for key, value in params.items():
            self.__dict__[key] = value
    
    @tf.function
    def call(self, features, labels, training=True):
        is_gpu_inference = not training and self.use_batched_nms
        batch_size, image_height, image_width, _ = features['images'].get_shape().as_list()
        #if 'source_ids' not in features:
        #    features['source_ids'] = -1 * tf.ones([batch_size], dtype=tf.float32)
        all_anchors = anchors.Anchors(self.min_level, self.max_level,
                                  self.num_scales, self.aspect_ratios,
                                  self.anchor_scale,
                                  (image_height, image_width))
        
        rpn_score_outputs, rpn_box_outputs, fpn_feats = self.bfr(features['images'], training=training)
        
        if training:
            rpn_pre_nms_topn = self.train_rpn_pre_nms_topn
            rpn_post_nms_topn = self.train_rpn_post_nms_topn
            rpn_nms_threshold = self.train_rpn_nms_threshold
        else:
            rpn_pre_nms_topn = self.test_rpn_pre_nms_topn
            rpn_post_nms_topn = self.test_rpn_post_nms_topn
            rpn_nms_threshold = self.test_rpn_nms_thresh
        
        rpn_box_scores, rpn_box_rois = roi_ops.custom_multilevel_propose_rois(
            scores_outputs=rpn_score_outputs,
            box_outputs=rpn_box_outputs,
            all_anchors=all_anchors,
            image_info=features['image_info'],
            rpn_pre_nms_topn=rpn_pre_nms_topn,
            rpn_post_nms_topn=rpn_post_nms_topn,
            rpn_nms_threshold=rpn_nms_threshold,
            rpn_min_size=self.rpn_min_size
        )
        
        model_outputs = self.rcnn(rpn_box_rois, rpn_box_scores, fpn_feats, 
                                  labels['gt_boxes'], features['image_info'],
                                  labels['gt_classes'], 
                                  is_gpu_inference, 
                                  cropped_gt_masks=labels.get('cropped_gt_masks'),
                                  training=training)
        model_outputs.update({
            'rpn_score_outputs': rpn_score_outputs,
            'rpn_box_outputs': rpn_box_outputs,
            })
        
        return model_outputs
    
    #@tf.function(experimental_compile=True)
    def bfr(self, images, training=True):
        backbone_feats = self.backbone(images, training=training)
        fpn_feats = self.fpn(backbone_feats, training=training)
        scores_outputs = dict()
        box_outputs = dict()
        for level in range(self.min_level, self.max_level + 1):
            scores_outputs[level], box_outputs[level] = self.rpn(fpn_feats[level], training=training)
        return scores_outputs, box_outputs, fpn_feats
    
    #@tf.function(experimental_compile=True)
    def rcnn(self, rpn_box_rois, rpn_box_scores, fpn_feats, gt_boxes, image_info,
             gt_classes, is_gpu_inference, cropped_gt_masks=None, training=True):
        model_outputs = {}
        rpn_box_rois = tf.cast(rpn_box_rois, dtype=tf.float32)
        if training:
            rpn_box_rois = tf.stop_gradient(rpn_box_rois)
            rpn_box_scores = tf.stop_gradient(rpn_box_scores)
            
            box_targets, class_targets, rpn_box_rois, proposal_to_label_map = training_ops.proposal_label_op(
                rpn_box_rois,
                gt_boxes,
                gt_classes,
                batch_size_per_im=self.batch_size_per_im,
                fg_fraction=self.fg_fraction,
                fg_thresh=self.fg_thresh,
                bg_thresh_hi=self.bg_thresh_hi,
                bg_thresh_lo=self.bg_thresh_lo
            )
        box_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats,
            boxes=rpn_box_rois,
            output_size=7,
            is_gpu_inference=is_gpu_inference
        )
        class_outputs, box_outputs, _ = self.box_head(inputs=box_roi_features)
        if not training:
            detections = postprocess_ops.generate_detections_gpu(
                class_outputs=class_outputs,
                box_outputs=box_outputs,
                anchor_boxes=rpn_box_rois,
                image_info=image_info,
                pre_nms_num_detections=self.test_rpn_post_nms_topn,
                post_nms_num_detections=self.test_detections_per_image,
                nms_threshold=self.test_nms,
                bbox_reg_weights=self.bbox_reg_weights
            )
            model_outputs.update({
                'num_detections': detections[0],
                'detection_boxes': detections[1],
                'detection_classes': detections[2],
                'detection_scores': detections[3],
            })
        else:
            encoded_box_targets = training_ops.encode_box_targets(
                boxes=rpn_box_rois,
                gt_boxes=box_targets,
                gt_labels=class_targets,
                bbox_reg_weights=self.bbox_reg_weights
            )
            model_outputs.update({
                'class_outputs': class_outputs,
                'box_outputs': box_outputs,
                'class_targets': class_targets,
                'box_targets': encoded_box_targets,
                'box_rois': rpn_box_rois,
            })
            
        if not self.include_mask:
            return model_outputs
        
        if not training:
            selected_box_rois = model_outputs['detection_boxes']
            class_indices = model_outputs['detection_classes']
        
        else:
            selected_class_targets, selected_box_targets, \
            selected_box_rois, proposal_to_label_map = training_ops.select_fg_for_masks(
                class_targets=class_targets,
                box_targets=box_targets,
                boxes=rpn_box_rois,
                proposal_to_label_map=proposal_to_label_map,
                max_num_fg=int(self.batch_size_per_im * self.fg_fraction)
                )
            class_indices = tf.cast(selected_class_targets, dtype=tf.int32)
            
        mask_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
                                                features=fpn_feats,
                                                boxes=selected_box_rois,
                                                output_size=14,
                                                is_gpu_inference=is_gpu_inference
                                            )
        mask_outputs = self.mask_head(inputs=mask_roi_features, class_indices=class_indices)
        
        if training:
            mask_targets = training_ops.get_mask_targets(fg_boxes=selected_box_rois,
                                                         fg_proposal_to_label_map=proposal_to_label_map,
                                                         fg_box_targets=selected_box_targets,
                                                         mask_gt_labels=cropped_gt_masks,
                                                         output_size=self.mrcnn_resolution
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