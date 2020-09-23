import os
import numpy as np
import sys
import itertools
from statistics import mean
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append('..')

import tensorflow as tf
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

import horovod.tensorflow as hvd
hvd.init()

physical_devices = tf.config.list_physical_devices('GPU')
# set two gpus visible per horovod rank
tf.config.set_visible_devices([physical_devices[hvd.rank()*2], 
                               physical_devices[hvd.rank()*2 + 1]], 'GPU')
devices = tf.config.list_logical_devices('GPU')

from mask_rcnn.tf2_parallel_model import MaskRCNN_Parallel, train_forward

from mask_rcnn.hyperparameters import dataset_params
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn import dataset_utils

train_file_pattern = '/home/ubuntu/data/coco/tf_record/train*'
batch_size = 1
data_params = dataset_params.get_data_params()
params = mask_rcnn_params.default_config().values()
data_params['batch_size'] = batch_size
params['finetune_bn'] = False
params['train_batch_size'] = batch_size
params['l2_weight_decay'] = 1e-4
params['init_learning_rate'] = 1e-3 * batch_size
params['warmup_learning_rate'] = 1e-4 * batch_size
params['warmup_steps'] = 500
params['learning_rate_steps'] = [30000,40000]
params['learning_rate_levels'] = [1e-4 * batch_size, 1e-5 * batch_size]
params['momentum'] = 0.9
params['use_batched_nms'] = True

train_input_fn = dataset_utils.FastDataLoader(train_file_pattern, data_params)
train_tdf = train_input_fn(data_params)

train_iter_0 = iter(train_tdf)
train_iter_1 = iter(train_tdf)

model = MaskRCNN_Parallel(params, devices)

features_0, labels_0 = next(train_iter_0)
features_1, labels_1 = next(train_iter_1)
model_outputs = model(features_0, features_1, labels_0, labels_1)

optimizer_0 = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
optimizer_0 = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer_0, 'dynamic')
optimizer_1 = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
optimizer_1 = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer_1, 'dynamic')

@tf.function
def train_step(features_0, features_1, labels_0, labels_1, model, devices, params):
    gpu_0_vars = [i for i in model.trainable_variables if 'GPU:0' in i.device]
    gpu_1_vars = [i for i in model.trainable_variables if 'GPU:1' in i.device]
    with tf.GradientTape(persistent=True) as tape:
        total_loss_0, total_loss_1 = train_forward(features_0, features_1, 
                                                   labels_0, labels_1, model, 
                                                   devices, params)
        scaled_total_loss_0 = optimizer_0.get_scaled_loss(total_loss_0)
        scaled_total_loss_1 = optimizer_1.get_scaled_loss(total_loss_1)
    scaled_gradients_0 = tape.gradient(scaled_total_loss_0, gpu_0_vars)
    scaled_gradients_1 = tape.gradient(scaled_total_loss_1, gpu_1_vars)
    gradients_0 = optimizer_0.get_unscaled_gradients(scaled_gradients_0)
    gradients_1 = optimizer_1.get_unscaled_gradients(scaled_gradients_1)
    optimizer_0.apply_gradients(zip(gradients_0, gpu_0_vars))
    optimizer_1.apply_gradients(zip(gradients_1, gpu_1_vars))
    return total_loss_0, total_loss_1

p_bar = tqdm(range(5000))
loss_history_0 = []
loss_history_1 = []
for i in p_bar:
    features_0, labels_0 = next(train_iter_0)
    features_1, labels_1 = next(train_iter_1)
    total_loss_0, total_loss_1 = train_step(features_0, features_1,
                                            labels_0, labels_1,
                                            model, devices, params)
    loss_history_0.append(total_loss_0.numpy())
    loss_history_1.append(total_loss_1.numpy())
    smoothed_loss_0 = mean(loss_history_0[-50:])
    smoothed_loss_1 = mean(loss_history_1[-50:])
    p_bar.set_description("Loss 0: {0:.4f}, Loss 1: {1:.4f}".format(smoothed_loss_0,
                                                                    smoothed_loss_1))
    
