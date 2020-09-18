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
tf.config.set_visible_devices([physical_devices[hvd.rank()*2], physical_devices[hvd.rank()*2+1]], 'GPU')
devices = tf.config.list_logical_devices('GPU')

from mask_rcnn.hyperparameters import dataset_params
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn import dataset_utils
from mask_rcnn.models.resnet import Resnet_Model
from simple_model import load_weights

# hardcoded hyperparameters
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

# split each image into half
def map0(features, labels):
    features['images'] = features['images'][:,:,:672+64,:]
    return features, labels

def map1(features, labels):
    features['images'] = features['images'][:,:,672-64:,:]
    return features, labels

# setup two pipelines from one source so the images match
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
images_0 = features_0['images']
images_1 = features_1['images']

def forward_single_gpu(images_0, images_1, is_training=True):
    resnet_0 = Resnet_Model(
                        "resnet50",
                        data_format='channels_last',
                        trainable=is_training,
                        finetune_bn=params['finetune_bn']
                    )
    resnet_1 = Resnet_Model(
                        "resnet50",
                        data_format='channels_last',
                        trainable=is_training,
                        finetune_bn=params['finetune_bn']
                    )
    features_0 = resnet_0(images_0)
    features_0[2] = features_0[2][:,:,:-16,:]
    features_0[3] = features_0[3][:,:,:-8,:]
    features_0[4] = features_0[4][:,:,:-4,:]
    features_0[5] = features_0[5][:,:,:-2,:]
    
    features_1 = resnet_1(images_1)
    features_1[2] = features_1[2][:,:,16:,:]
    features_1[3] = features_1[3][:,:,8:,:]
    features_1[4] = features_1[4][:,:,4:,:]
    features_1[5] = features_1[5][:,:,2:,:]
    
    C2_0_1 = tf.identity(features_0[2])
    C2_1_0 = tf.identity(features_1[2])
    
    C3_0_1 = tf.identity(features_0[3])
    C3_1_0 = tf.identity(features_1[3])
    
    C4_0_1 = tf.identity(features_0[4])
    C4_1_0 = tf.identity(features_1[4])
    
    C5_0_1 = tf.identity(features_0[5])
    C5_1_0 = tf.identity(features_1[5])
    
    features_0[2] = tf.concat([features_0[2], C2_1_0], axis=2)
    features_0[3] = tf.concat([features_0[3], C3_1_0], axis=2)
    features_0[4] = tf.concat([features_0[4], C4_1_0], axis=2)
    features_0[5] = tf.concat([features_0[5], C5_1_0], axis=2)
    
    features_1[2] = tf.concat([C2_0_1, features_1[2]], axis=2)
    features_1[3] = tf.concat([C3_0_1, features_1[3]], axis=2)
    features_1[4] = tf.concat([C4_0_1, features_1[4]], axis=2)
    features_1[5] = tf.concat([C5_0_1, features_1[5]], axis=2)
    
    return features_0, features_1

def forward_2_gpu(images_0, images_1, is_training=True):
    
    # Place ops on each device
    with tf.device(devices[0].name):
        resnet_0 = Resnet_Model(
                            "resnet50",
                            data_format='channels_last',
                            trainable=is_training,
                            finetune_bn=params['finetune_bn']
                        )
    with tf.device(devices[1].name):
        resnet_1 = Resnet_Model(
                            "resnet50",
                            data_format='channels_last',
                            trainable=is_training,
                            finetune_bn=params['finetune_bn']
                        )
    # Run half of resnet on each
    with tf.device(devices[0].name):
        features_0 = resnet_0(images_0)
        features_0[2] = features_0[2][:,:,:-16,:]
        features_0[3] = features_0[3][:,:,:-8,:]
        features_0[4] = features_0[4][:,:,:-4,:]
        features_0[5] = features_0[5][:,:,:-2,:]
    
    with tf.device(devices[1].name):
        features_1 = resnet_1(images_1)
        features_1[2] = features_1[2][:,:,16:,:]
        features_1[3] = features_1[3][:,:,8:,:]
        features_1[4] = features_1[4][:,:,4:,:]
        features_1[5] = features_1[5][:,:,2:,:]
    
    # copy tensors from device 1 to device 0 cut off overlapping section
    with tf.device(devices[0].name):
        C2_1_0 = tf.stop_gradient(tf.identity(features_1[2]))
        C3_1_0 = tf.stop_gradient(tf.identity(features_1[3]))
        C4_1_0 = tf.stop_gradient(tf.identity(features_1[4]))
        C5_1_0 = tf.stop_gradient(tf.identity(features_1[5]))
    
    # copy tensors from device 0 to device 1 cut off overlapping section
    with tf.device(devices[1].name):
        C2_0_1 = tf.stop_gradient(tf.identity(features_0[2]))
        C3_0_1 = tf.stop_gradient(tf.identity(features_0[3]))
        C4_0_1 = tf.stop_gradient(tf.identity(features_0[4]))
        C5_0_1 = tf.stop_gradient(tf.identity(features_0[5]))
    
    # concatenate tensors
    with tf.device(devices[0].name):
        features_0[2] = tf.concat([features_0[2], C2_1_0], axis=2)
        features_0[3] = tf.concat([features_0[3], C3_1_0], axis=2)
        features_0[4] = tf.concat([features_0[4], C4_1_0], axis=2)
        features_0[5] = tf.concat([features_0[5], C5_1_0], axis=2)
    
    with tf.device(devices[1].name):
        features_1[2] = tf.concat([C2_0_1, features_1[2]], axis=2)
        features_1[3] = tf.concat([C3_0_1, features_1[3]], axis=2)
        features_1[4] = tf.concat([C4_0_1, features_1[4]], axis=2)
        features_1[5] = tf.concat([C5_0_1, features_1[5]], axis=2)
    
    return features_0, features_1

resnet_outputs = forward_2_gpu(images_0, images_1)

# load pretrained weights to both backbones
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
    p_bar = tqdm(range(5000))
    for i in p_bar:
        output = sess.run(resnet_outputs)