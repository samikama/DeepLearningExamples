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

"""Losses used for Mask-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion

import tensorflow as tf
from mask_rcnn.utils import box_utils
import math 

DEBUG_LOSS_IMPLEMENTATION = False


if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
    from tensorflow.python.keras.utils import losses_utils
    ReductionV2 = losses_utils.ReductionV2
else:
    ReductionV2 = tf.keras.losses.Reduction

def _calculate_giou(b1, b2, mode="giou"):
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.
    Returns:
        GIoU loss float `Tensor`.
    """
    zero = tf.convert_to_tensor(0.0, b1.dtype)
    b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == "iou":
        return iou

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
    return giou


def _calculate_ciou(b1, b2, mode="diou"):
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['diou', 'ciou'], decided to calculate GIoU or IoU loss.
    Returns:
        GIoU loss float `Tensor`.
    """
    zero = tf.convert_to_tensor(0.0, b1.dtype)

    b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == "iou":
        return iou

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height

    if mode == "giou":
        giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
        return giou

    # CIoU - https://arxiv.org/pdf/1911.08287.pdf
    diag_length = tf.linalg.norm([enclose_height, enclose_width]) # tf.math.square(enclose_width) + tf.math.square(enclose_height)
    b1_center = tf.stack([(b1_ymin + b1_ymax) / 2., (b1_xmin + b1_xmax) / 2.])
    b2_center = tf.stack([(b2_ymin + b2_ymax) / 2., (b2_xmin + b2_xmax) / 2.])
    centers_dist = tf.linalg.norm([b1_center-b2_center])

    diou = iou - tf.math.divide_no_nan(centers_dist**2, diag_length**2)

    if mode == "diou":
        return diou

    arctan = tf.atan(tf.math.divide_no_nan(b1_width, b1_height)) - tf.atan(tf.math.divide_no_nan(b2_width, b2_height))
    v = 4.0 * ((arctan / math.pi) ** 2)

    # apply aspect ratio penalty only if IoU > 0.5 and GT box is in medium or large category
    # aspect_penalty_mask = tf.cast(tf.math.logical_and(iou > 0.5, b1_area > 1024.), b1.dtype) # don't know size of box after resize!
    aspect_penalty_mask = tf.cast(iou > 0.5, b1.dtype)
    alpha = aspect_penalty_mask * tf.math.divide_no_nan(v, 1.0 - iou + v)

    ciou = diou - alpha * v

    return ciou



def _giou_loss(y_true, y_pred, weights):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    weights = tf.cast(weights, tf.float32)
    y_true = tf.reshape(y_true, [-1, 4])
    y_pred = tf.reshape(y_pred, [-1, 4])
    weights = tf.reshape(weights, [-1, 4])
    giou = _calculate_giou(y_true, y_pred)
    giou_loss = 1. - giou
    giou_loss = tf.tile(tf.expand_dims(giou_loss, -1), [1, 4]) * weights # only take pos example contributions
    giou_loss = tf.math.divide_no_nan(tf.math.reduce_sum(giou_loss), tf.math.count_nonzero(weights, dtype=tf.float32))
    assert giou_loss.dtype == tf.float32
    return giou_loss


def _ciou_loss(y_true, y_pred, weights):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    weights = tf.cast(weights, tf.float32)
    y_true = tf.reshape(y_true, [-1, 4])
    y_pred = tf.reshape(y_pred, [-1, 4])
    weights = tf.reshape(weights, [-1, 4])
    ciou = _calculate_ciou(b1=y_true, b2=y_pred, mode="ciou")
    ciou_loss = 1. - ciou

    ciou_loss = tf.tile(tf.expand_dims(ciou_loss, -1), [1, 4]) * weights # only take pos example contributions
    ciou_loss = tf.math.divide_no_nan(tf.math.reduce_sum(ciou_loss), tf.math.count_nonzero(weights, dtype=tf.float32))
    assert ciou_loss.dtype == tf.float32
    return ciou_loss


def _l1_loss(y_true, y_pred, weights, delta=0.0):
    l1_loss = tf.compat.v1.losses.absolute_difference(y_true, y_pred, weights=weights)
    assert l1_loss.dtype == tf.float32
    DEBUG_LOSS_IMPLEMENTATION = False
    if DEBUG_LOSS_IMPLEMENTATION:
        mlperf_loss = tf.compat.v1.losses.huber_loss(
            y_true,
            y_pred,
            weights=weights,
            delta=delta,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )

        print_op = tf.print("Huber Loss - MLPerf:", mlperf_loss, " && Legacy Loss:", l1_loss)

        with tf.control_dependencies([print_op]):
            l1_loss = tf.identity(l1_loss)

    return l1_loss


def _huber_loss(y_true, y_pred, weights, delta):

    num_non_zeros = tf.math.count_nonzero(weights, dtype=tf.float32)

    huber_keras_loss = tf.keras.losses.Huber(
        delta=delta,
        reduction=ReductionV2.SUM,
        name='huber_loss'
    )

    if LooseVersion(tf.__version__) >= LooseVersion("2.2.0"):
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)

    huber_loss = huber_keras_loss(
        y_true,
        y_pred,
        sample_weight=weights
    )

    assert huber_loss.dtype == tf.float32

    huber_loss = tf.math.divide_no_nan(huber_loss, num_non_zeros, name="huber_loss")

    assert huber_loss.dtype == tf.float32

    if DEBUG_LOSS_IMPLEMENTATION:
        mlperf_loss = tf.compat.v1.losses.huber_loss(
            y_true,
            y_pred,
            weights=weights,
            delta=delta,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )

        print_op = tf.print("Huber Loss - MLPerf:", mlperf_loss, " && Legacy Loss:", huber_loss)

        with tf.control_dependencies([print_op]):
            huber_loss = tf.identity(huber_loss)

    return huber_loss


def _sigmoid_cross_entropy(multi_class_labels, logits, weights, sum_by_non_zeros_weights=False, label_smoothing=0.0):

    assert weights.dtype == tf.float32

#    sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
#        labels=multi_class_labels,
#        logits=logits,
#        name="x-entropy"
#    )

    sigmoid_cross_entropy = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=multi_class_labels,
        logits=logits,
        label_smoothing=label_smoothing,
        reduction=tf.compat.v1.losses.Reduction.NONE
    )

    assert sigmoid_cross_entropy.dtype == tf.float32

    sigmoid_cross_entropy = tf.math.multiply(sigmoid_cross_entropy, weights)
    sigmoid_cross_entropy = tf.math.reduce_sum(sigmoid_cross_entropy)

    assert sigmoid_cross_entropy.dtype == tf.float32

    if sum_by_non_zeros_weights:
        num_non_zeros = tf.math.count_nonzero(weights, dtype=tf.float32)
        sigmoid_cross_entropy = tf.math.divide_no_nan(
            sigmoid_cross_entropy,
            num_non_zeros,
            name="sum_by_non_zeros_weights"
        )

    assert sigmoid_cross_entropy.dtype == tf.float32

    if DEBUG_LOSS_IMPLEMENTATION:

        if sum_by_non_zeros_weights:
            reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

        else:
            reduction = tf.compat.v1.losses.Reduction.SUM

        mlperf_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            multi_class_labels=multi_class_labels,
            logits=logits,
            weights=weights,
            reduction=reduction
        )

        print_op = tf.print(
            "Sigmoid X-Entropy Loss (%s) - MLPerf:" % reduction, mlperf_loss, " && Legacy Loss:", sigmoid_cross_entropy
        )

        with tf.control_dependencies([print_op]):
            sigmoid_cross_entropy = tf.identity(sigmoid_cross_entropy)

    return sigmoid_cross_entropy


def _softmax_cross_entropy(onehot_labels, logits, label_smoothing=0.0):

    num_non_zeros = tf.math.count_nonzero(onehot_labels, dtype=tf.float32)
    if label_smoothing == 0.0:
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=onehot_labels,
            logits=logits
        )
    else:
        softmax_cross_entropy = tf.compat.v1.losses.softmax_cross_entropy(
            onehot_labels,
            logits,
            label_smoothing=label_smoothing,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )

    assert softmax_cross_entropy.dtype == tf.float32

    if label_smoothing == 0.0:
        softmax_cross_entropy = tf.math.reduce_sum(softmax_cross_entropy)
        softmax_cross_entropy = tf.math.divide_no_nan(softmax_cross_entropy, num_non_zeros, name="softmax_cross_entropy")

    assert softmax_cross_entropy.dtype == tf.float32

    DEBUG_LOSS_IMPLEMENTATION = False

    if DEBUG_LOSS_IMPLEMENTATION:

        mlperf_loss = tf.compat.v1.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )

        print_op = tf.print("Softmax X-Entropy Loss - MLPerf:", mlperf_loss, " && Legacy Loss:", softmax_cross_entropy)

        with tf.control_dependencies([print_op]):
            softmax_cross_entropy = tf.identity(softmax_cross_entropy)

    return softmax_cross_entropy


def _rpn_score_loss(score_outputs, score_targets, normalizer=1.0, label_smoothing=0.0):
    """Computes score loss."""

    with tf.name_scope('rpn_score_loss'):

        # score_targets has three values:
        # * (1) score_targets[i]=1, the anchor is a positive sample.
        # * (2) score_targets[i]=0, negative.
        # * (3) score_targets[i]=-1, the anchor is don't care (ignore).

        mask = tf.math.greater_equal(score_targets, 0)
        mask = tf.cast(mask, dtype=tf.float32)

        score_targets = tf.maximum(score_targets, tf.zeros_like(score_targets))
        score_targets = tf.cast(score_targets, dtype=tf.float32)

        assert score_outputs.dtype == tf.float32
        assert score_targets.dtype == tf.float32

        score_loss = _sigmoid_cross_entropy(
            multi_class_labels=score_targets,
            logits=score_outputs,
            weights=mask,
            sum_by_non_zeros_weights=False,
            label_smoothing=label_smoothing
        )

        assert score_loss.dtype == tf.float32

        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            score_loss /= normalizer

        assert score_loss.dtype == tf.float32

    return score_loss


def _rpn_box_loss(box_outputs, box_targets, normalizer=1.0, delta=1. / 9):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].

    with tf.name_scope('rpn_box_loss'):
        mask = tf.not_equal(box_targets, 0.0)
        mask = tf.cast(mask, tf.float32)

        assert mask.dtype == tf.float32

        # The loss is normalized by the sum of non-zero weights before additional
        # normalizer provided by the function caller.
        box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)
        # box_loss = _l1_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)
        
        assert box_loss.dtype == tf.float32

        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            box_loss /= normalizer

        assert box_loss.dtype == tf.float32

    return box_loss


def rpn_loss(score_outputs, box_outputs, labels, params):
    """Computes total RPN detection loss.

    Computes total RPN detection loss including box and score from all levels.
    Args:
    score_outputs: an OrderDict with keys representing levels and values
      representing scores in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundturth targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
    Returns:
    total_rpn_loss: a float tensor representing total loss reduced from
      score and box losses from all levels.
    rpn_score_loss: a float tensor representing total score loss.
    rpn_box_loss: a float tensor representing total box regression loss.
    """
    with tf.name_scope('rpn_loss'):

        score_losses = []
        box_losses = []

        for level in range(int(params['min_level']), int(params['max_level'] + 1)):

            score_targets_at_level = labels['score_targets_%d' % level]
            box_targets_at_level = labels['box_targets_%d' % level]

            score_losses.append(
                _rpn_score_loss(
                    score_outputs=score_outputs[level],
                    score_targets=score_targets_at_level,
                    normalizer=tf.cast(params['train_batch_size'] * params['rpn_batch_size_per_im'], dtype=tf.float32),
                    label_smoothing=params['label_smoothing']
                )
            )

            box_losses.append(_rpn_box_loss(
                box_outputs=box_outputs[level],
                box_targets=box_targets_at_level,
                normalizer=1.0
            ))

        # Sum per level losses to total loss.
        rpn_score_loss = tf.add_n(score_losses)
        rpn_box_loss = params['rpn_box_loss_weight'] * tf.add_n(box_losses)

        total_rpn_loss = rpn_score_loss + rpn_box_loss

    return total_rpn_loss, rpn_score_loss, rpn_box_loss


def _fast_rcnn_class_loss(class_outputs, class_targets_one_hot, normalizer=1.0):
    """Computes classification loss."""

    with tf.name_scope('fast_rcnn_class_loss'):
        # The loss is normalized by the sum of non-zero weights before additional
        # normalizer provided by the function caller.

        class_loss = _softmax_cross_entropy(onehot_labels=class_targets_one_hot, logits=class_outputs)

        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            class_loss /= normalizer

    return class_loss


def _fast_rcnn_box_loss(box_outputs, box_targets, class_targets, loss_type='huber', normalizer=1.0, delta=1.):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].

    with tf.name_scope('fast_rcnn_box_loss'):
        mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2), [1, 1, 4])

        # The loss is normalized by the sum of non-zero weights before additional
        # normalizer provided by the function caller.
        if loss_type == 'huber':
            box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)
        elif loss_type == 'giou':
            box_loss = _giou_loss(y_true=box_targets, y_pred=box_outputs, weights=mask)
        elif loss_type == 'ciou':
            box_loss = _ciou_loss(y_true=box_targets, y_pred=box_outputs, weights=mask)
        else:
            # box_loss = _l1_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)
            raise NotImplementedError
        
        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            box_loss /= normalizer

    return box_loss


def fast_rcnn_loss(class_outputs, box_outputs, class_targets, box_targets, rpn_box_rois, image_info, params):
    """Computes the box and class loss (Fast-RCNN branch) of Mask-RCNN.

    This function implements the classification and box regression loss of the
    Fast-RCNN branch in Mask-RCNN. As the `box_outputs` produces `num_classes`
    boxes for each RoI, the reference model expands `box_targets` to match the
    shape of `box_outputs` and selects only the target that the RoI has a maximum
    overlap. (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py)
    Instead, this function selects the `box_outputs` by the `class_targets` so
    that it doesn't expand `box_targets`.

    The loss computation has two parts: (1) classification loss is softmax on all
    RoIs. (2) box loss is smooth L1-loss on only positive samples of RoIs.
    Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py


    Args:
    class_outputs: a float tensor representing the class prediction for each box
      with a shape of [batch_size, num_boxes, num_classes].
    box_outputs: a float tensor representing the box prediction for each box
      with a shape of [batch_size, num_boxes, num_classes * 4].
    class_targets: a float tensor representing the class label for each box
      with a shape of [batch_size, num_boxes].
    box_targets: a float tensor representing the box label for each box
      with a shape of [batch_size, num_boxes, 4].
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
    Returns:
    total_loss: a float tensor representing total loss reducing from
      class and box losses from all levels.
    cls_loss: a float tensor representing total class loss.
    box_loss: a float tensor representing total box regression loss.
    """
    with tf.name_scope('fast_rcnn_loss'):
        class_targets = tf.cast(class_targets, dtype=tf.int32)

        # Selects the box from `box_outputs` based on `class_targets`, with which
        # the box has the maximum overlap.
        batch_size, num_rois, _ = box_outputs.get_shape().as_list()
        box_outputs = tf.reshape(box_outputs, [batch_size, num_rois, params['num_classes'], 4])

        box_indices = tf.reshape(
            class_targets +
            tf.tile(tf.expand_dims(tf.range(batch_size) * num_rois * params['num_classes'], 1), [1, num_rois]) +
            tf.tile(tf.expand_dims(tf.range(num_rois) * params['num_classes'], 0), [batch_size, 1]),
            [-1]
        )

        box_outputs = tf.matmul(
            tf.one_hot(
                box_indices,
                batch_size * num_rois * params['num_classes'],
                dtype=box_outputs.dtype
            ),
            tf.reshape(box_outputs, [-1, 4])
        )

        if params['box_loss_type'] in ['giou', 'ciou']:
            # decode outputs to move deltas back to coordinate space
            rpn_box_rois = tf.reshape(rpn_box_rois, [-1, 4])
            box_outputs = box_utils.decode_boxes(encoded_boxes=box_outputs, anchors=rpn_box_rois, weights=params['bbox_reg_weights'])
            # Clip boxes FIXME: hardcoding for now
            box_outputs = box_utils.clip_boxes(box_outputs, 832., 1344.)


        box_outputs = tf.reshape(box_outputs, [batch_size, -1, 4])
        box_loss = _fast_rcnn_box_loss(
            box_outputs=box_outputs,
            box_targets=box_targets,
            class_targets=class_targets,
            loss_type=params['box_loss_type'],
            normalizer=1.0
        )
        box_loss *= params['fast_rcnn_box_loss_weight']

        use_sparse_x_entropy = False

        _class_targets = class_targets if use_sparse_x_entropy else tf.one_hot(class_targets, params['num_classes'])

        class_loss = _fast_rcnn_class_loss(
            class_outputs=class_outputs,
            class_targets_one_hot=_class_targets,
            normalizer=1.0
        )

        total_loss = class_loss + box_loss

    return total_loss, class_loss, box_loss


def mask_rcnn_loss(mask_outputs, mask_targets, select_class_targets, params):
    """Computes the mask loss of Mask-RCNN.

    This function implements the mask loss of Mask-RCNN. As the `mask_outputs`
    produces `num_classes` masks for each RoI, the reference model expands
    `mask_targets` to match the shape of `mask_outputs` and selects only the
    target that the RoI has a maximum overlap.
    (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py)
    Instead, this implementation selects the `mask_outputs` by the `class_targets`
    so that it doesn't expand `mask_targets`. Note that the selection logic is
    done in the post-processing of mask_rcnn_fn in mask_rcnn_architecture.py.

    Args:
    mask_outputs: a float tensor representing the prediction for each mask,
      with a shape of
      [batch_size, num_masks, mask_height, mask_width].
    mask_targets: a float tensor representing the binary mask of ground truth
      labels for each mask with a shape of
      [batch_size, num_masks, mask_height, mask_width].
    select_class_targets: a tensor with a shape of [batch_size, num_masks],
      representing the foreground mask targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
    Returns:
    mask_loss: a float tensor representing total mask loss.
    """
    with tf.name_scope('mask_loss'):
        batch_size, num_masks, mask_height, mask_width = mask_outputs.get_shape().as_list()

        weights = tf.tile(
            tf.reshape(tf.greater(select_class_targets, 0), [batch_size, num_masks, 1, 1]),
            [1, 1, mask_height, mask_width]
        )
        weights = tf.cast(weights, tf.float32)

        loss = _sigmoid_cross_entropy(
            multi_class_labels=mask_targets,
            logits=mask_outputs,
            weights=weights,
            sum_by_non_zeros_weights=True,
            label_smoothing=params['label_smoothing']
        )

        mrcnn_loss = params['mrcnn_weight_loss_mask'] * loss

        return mrcnn_loss
