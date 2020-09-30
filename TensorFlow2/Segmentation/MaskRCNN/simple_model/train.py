import os
import sys
sys.path.append('..')
import itertools
from time import time
from statistics import mean
os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=fusible"
#os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
from tqdm import tqdm
import numpy as np
sys.path.append('..')
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()
import horovod.tensorflow as hvd
hvd.init()

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[hvd.rank()], 'GPU')

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

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
import model
import load_weights

train_file_pattern = '/home/ubuntu/data/coco/tf_record/train*'
batch_size = 4

data_params = dataset_params.get_data_params()
params = mask_rcnn_params.default_config().values()

data_params['batch_size'] = batch_size
params['finetune_bn'] = False
params['train_batch_size'] = batch_size
params['l2_weight_decay'] = 1e-4
params['init_learning_rate'] = 1e-4 * batch_size * hvd.size()
params['warmup_learning_rate'] = 1e-3 * batch_size * hvd.size()
params['warmup_steps'] = 500
params['learning_rate_steps'] = [30000,40000]
params['learning_rate_levels'] = [1e-4 * batch_size * hvd.size(), 1e-5 * batch_size * hvd.size()]
params['momentum'] = 0.9
params['use_batched_nms'] = True

train_input_fn = dataset_utils.FastDataLoader(train_file_pattern, data_params)
train_tdf = train_input_fn(data_params)

tdf_iter = train_tdf.make_initializable_iterator()
features, labels = tdf_iter.get_next()

train_op, total_loss = model.model(features, params, labels)

var_list = load_weights.build_assigment_map('resnet50/')
checkpoint_file = tf.train.latest_checkpoint('../resnet/resnet-nhwc-2018-02-07/')
_init_op, _init_feed_dict = load_weights.assign_from_checkpoint(checkpoint_file, var_list)

steps = 118000//(batch_size * hvd.size())

var_initializer = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(_init_op, _init_feed_dict)
    sess.run(tdf_iter.initializer)
    sess.run(var_initializer)
    for epoch in range(12):
        if hvd.rank()==0:
            progressbar = tqdm(range(steps))
            loss_history = []
        else:
            progressbar = range(steps)
        for i in progressbar:
            op, loss = sess.run((train_op, total_loss))
            if hvd.rank()==0:
                loss_history.append(loss)
                progressbar.set_description("Loss: {0:.4f}".format(np.array(loss_history[-50:]).mean()))
