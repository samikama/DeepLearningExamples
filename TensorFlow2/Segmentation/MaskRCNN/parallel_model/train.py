import os
import sys
import itertools
from statistics import mean
from time import time
from tqdm import tqdm
sys.path.append('..')
os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=fusible"
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

import horovod.tensorflow as hvd
hvd.init()

physical_devices = tf.config.list_physical_devices('GPU')
# set two gpus visible per horovod rank
tf.config.set_visible_devices([physical_devices[hvd.rank()*2], 
                               physical_devices[hvd.rank()*2 + 1]], 'GPU')
devices = tf.config.list_logical_devices('GPU')

from mask_rcnn.hyperparameters import dataset_params
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn import dataset_utils
from simple_model import load_weights

train_file_pattern = '/home/ubuntu/data/coco/tf_record/train*'
batch_size = 1
data_params = dataset_params.get_data_params()
params = mask_rcnn_params.default_config().values()
data_params['batch_size'] = batch_size
params['finetune_bn'] = False
params['train_batch_size'] = batch_size
params['l2_weight_decay'] = 1e-4
params['init_learning_rate'] = 1e-4 * batch_size
params['warmup_learning_rate'] = 1e-3 * batch_size
params['warmup_steps'] = 500
params['learning_rate_steps'] = [30000,40000]
params['learning_rate_levels'] = [1e-4 * batch_size, 1e-5 * batch_size]
params['momentum'] = 0.9
params['use_batched_nms'] = True

def map0(features, labels):
    features['images'] = features['images'][:,:,:672+64,:]
    return features, labels

def map1(features, labels):
    features['images'] = features['images'][:,:,672-64:,:]
    return features, labels

# create two data pipelines for each half of an image
# this seems to be pretty CPU intensive, should find
# another way to do this
train_input_fn = dataset_utils.FastDataLoader(train_file_pattern, data_params)
train_tdf = train_input_fn(data_params)
train_tdf_0 = train_tdf.map(map0, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_tdf_1 = train_tdf.map(map1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_tdf_0 = train_tdf_0.apply(tf.data.experimental.prefetch_to_device(devices[0].name))
train_tdf_1 = train_tdf_1.apply(tf.data.experimental.prefetch_to_device(devices[1].name))
tdf_iter_0 = train_tdf_0.make_initializable_iterator()
tdf_iter_1 = train_tdf_1.make_initializable_iterator()
features_0, labels_0 = tdf_iter_0.get_next()
features_1, labels_1 = tdf_iter_1.get_next()

import model_v2 as model


train_op_0, train_op_1, total_loss_0, total_loss_1 = model.model(features_0, 
                                                                 features_1, 
                                                                 params, 
                                                                 devices, 
                                                                 labels_0, 
                                                                 labels_1)

var_list_0 = load_weights.build_assigment_map('resnet50/')
var_list_1 = load_weights.build_assigment_map('resnet50_1/')
checkpoint_file = tf.train.latest_checkpoint('../resnet/resnet-nhwc-2018-02-07/')
_init_op_0, _init_feed_dict_0 = load_weights.assign_from_checkpoint(checkpoint_file, var_list_0)
_init_op_1, _init_feed_dict_1 = load_weights.assign_from_checkpoint(checkpoint_file, var_list_1)

var_initializer = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(_init_op_0, _init_feed_dict_0)
    sess.run(_init_op_1, _init_feed_dict_1)
    sess.run(tdf_iter_0.initializer)
    sess.run(tdf_iter_1.initializer)
    sess.run(var_initializer)
    loss_history_0 = []
    loss_history_1 = []
    p_bar = tqdm(range(10000))
    for i in p_bar:
        _, _, loss_0, loss_1 = sess.run([train_op_0, train_op_1, total_loss_0, total_loss_1])
        loss_history_0.append(loss_0)
        loss_history_1.append(loss_1)
        ma_loss_0 = mean(loss_history_0[-100:])
        ma_loss_1 = mean(loss_history_1[-100:])
        p_bar.set_description("Loss 0: {0:.4f}, Loss 1: {1:.4f}".format(ma_loss_0, ma_loss_1))
