import os
import sys
sys.path.append('..')
os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=fusible"
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import horovod.tensorflow as hvd
hvd.init()

physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[hvd.rank()], True)
tf.config.set_visible_devices(physical_devices[hvd.rank()], 'GPU')
devices = tf.config.list_logical_devices('GPU')

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)

from tqdm import tqdm
from statistics import mean

from mask_rcnn.tf2_model import MaskRCNN
from mask_rcnn.hyperparameters import dataset_params
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn import dataset_utils
from mask_rcnn.training import losses, learning_rates
from simple_model.tf2 import weight_loader, train, scheduler
from simple_model import model_v2

train_file_pattern = '/home/ubuntu/data/coco/tf_record/train*'
batch_size = 1
images = 118287
steps_per_epoch = images//(batch_size * hvd.size())
data_params = dataset_params.get_data_params()
params = mask_rcnn_params.default_config().values()
data_params['batch_size'] = batch_size
params['finetune_bn'] = False
params['train_batch_size'] = batch_size
params['l2_weight_decay'] = 1e-4
params['init_learning_rate'] = 2e-3 * batch_size * hvd.size()
params['warmup_learning_rate'] = 2e-4 * batch_size * hvd.size()
params['warmup_steps'] = 2000//hvd.size()
params['learning_rate_steps'] = [steps_per_epoch * 9, steps_per_epoch * 11]
params['learning_rate_levels'] = [2e-4 * batch_size, 2e-5 * batch_size]
params['momentum'] = 0.9
params['use_batched_nms'] = False
params['use_custom_box_proposals_op'] = True
params['amp'] = True
params['include_groundtruth_in_features'] = True

loader = dataset_utils.FastDataLoader(train_file_pattern, data_params)
train_tdf = loader(data_params)
train_tdf = train_tdf.apply(tf.data.experimental.prefetch_to_device(devices[0].name, 
                                                                    buffer_size=tf.data.experimental.AUTOTUNE))
train_iter = iter(train_tdf)

mask_rcnn = model_v2.MRCNN(params)

features, labels = next(train_iter)
model_outputs = mask_rcnn(features, labels, params, is_training=True)

weight_loader.load_resnet_checkpoint(mask_rcnn, '../resnet/resnet-nhwc-2018-02-07/')

schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(params['learning_rate_steps'],
                                                                [params['init_learning_rate']] \
                                                                + params['learning_rate_levels'])
schedule = scheduler.WarmupScheduler(schedule, params['warmup_learning_rate'],
                                     params['warmup_steps'])
optimizer = tf.keras.optimizers.SGD(schedule, momentum=0.9)
optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')

@tf.function
def train_step(features, labels, params, model, opt, first=False):
    with tf.GradientTape() as tape:
        total_loss = train.train_forward(features, labels, params, model)
        scaled_loss = optimizer.get_scaled_loss(total_loss)
    tape = hvd.DistributedGradientTape(tape)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if first:
        hvd.broadcast_variables(model.variables, 0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
    return total_loss

_ = train_step(features, labels, params, mask_rcnn, optimizer, first=True)

if hvd.rank()==0:
    p_bar = tqdm(range(steps_per_epoch))
    loss_history = []
else:
    p_bar = range(steps_per_epoch)
for i in p_bar:
    features, labels = next(train_iter)
    total_loss = train_step(features, labels, params, mask_rcnn, optimizer)
    if hvd.rank()==0:
        loss_history.append(total_loss.numpy())
        smoothed_loss = mean(loss_history[-50:])
        p_bar.set_description("Loss: {0:.4f}, LR: {1:.4f}".format(smoothed_loss, 
                                                                  schedule(optimizer.iterations)))
        