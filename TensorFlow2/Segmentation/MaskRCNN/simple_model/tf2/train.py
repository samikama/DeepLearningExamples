import tensorflow as tf
import sys
sys.path.append('../..')
from mask_rcnn.training import losses, learning_rates

def train_forward(features, labels, params, model):
    model_outputs = model(features, labels, params, is_training=True)
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
        params=params
    )

    # Only training has the mask loss.
    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/model_builder.py
    mask_loss = losses.mask_rcnn_loss(
            mask_outputs=model_outputs['mask_outputs'],
            mask_targets=model_outputs['mask_targets'],
            select_class_targets=model_outputs['selected_class_targets'],
            params=params
        )

    trainable_variables = model.trainable_variables

    l2_regularization_loss = params['l2_weight_decay'] * tf.add_n([
        tf.nn.l2_loss(v)
        for v in trainable_variables
        if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
    ])

    total_loss = total_rpn_loss + total_fast_rcnn_loss + mask_loss + l2_regularization_loss
    return total_loss