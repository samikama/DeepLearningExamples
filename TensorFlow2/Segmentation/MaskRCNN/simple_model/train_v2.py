import os
from mpi4py import MPI

os.environ["CUDA_VISIBLE_DEVICES"]=str(MPI.COMM_WORLD.Get_rank())
import numpy as np
import sys
import itertools
from statistics import mean
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sys.path.append('..')

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "1"
os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=fusible"
os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=1"
#os.environ["TF_CPP_VMODULE"] = 'roi_align_op=2'
do_profile=False


os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ADJUST_HUE_FUSED'] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
import tensorflow as tf
from tensorflow.python.profiler import profiler_v2 as tf_profiler
from tensorflow.python.profiler.trace import Trace as prof_Trace


from tensorflow.keras.mixed_precision import experimental as mixed_precision


import horovod.tensorflow as hvd
hvd.init()
physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[hvd.rank()], True)
#tf.config.set_visible_devices(physical_devices[hvd.rank()], 'GPU')
devices = tf.config.list_logical_devices('GPU')

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
from mask_rcnn.tf2_model import MaskRCNN
from mask_rcnn.hyperparameters import dataset_params
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn import dataset_utils
from mask_rcnn.training import losses, learning_rates
from simple_model.tf2 import weight_loader, train, scheduler
from simple_model import model_v2

batch_size = 1
images = 118287
steps_per_epoch = images//(batch_size * hvd.size())
data_params = dataset_params.get_data_params()
params = mask_rcnn_params.default_config().values()

train_file_pattern = '/data/coco/coco-2017/tfr_anchor/train*'
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
params['use_default_roi_align'] = False
params['use_transposed_features'] = True

loader = dataset_utils.FastDataLoader(train_file_pattern, data_params)
train_tdf = loader(data_params)
train_tdf = train_tdf.apply(tf.data.experimental.prefetch_to_device(devices[0].name, 
                                                                    buffer_size=tf.data.experimental.AUTOTUNE))
train_iter = iter(train_tdf)

mask_rcnn = model_v2.MRCNN(params)
features, labels = next(train_iter)
model_outputs = mask_rcnn(features, labels, params, is_training=True)

weight_loader.load_resnet_checkpoint(mask_rcnn, '../weights/resnet/resnet-nhwc-2018-02-07/')

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
def get_env_value(env_str, key):
  pos = env_str.find(key)
  if pos >= 0:
    substr = env_str[pos + len(key)]
    if substr.find(" ") == -1:
      return substr
    else:
      return substr[:substr.find(" ")]
  return None
xlaflags = os.environ.get("TF_XLA_FLAGS", "")
suffix=""
if not params["use_default_roi_align"]:
  suffix+="_roi"
  suffix+= "_chw" if params["use_transposed_features"] else "_hwc"

if "fusible" in xlaflags:
  suffix += "_fusible"
elif "tf_xla_auto_jit=" in xlaflags:
  val = get_env_value(xlaflags, "tf_xla_auto_jit=")
  if val:
    suffix += f"_jit_{val}"

if "min_cluster_size" in xlaflags:
  val = get_env_value(xlaflags, "min_cluster_size=")
  if val:
    suffix += "_csize" + val


if os.environ.get("TF_GPU_THREAD_MODE") == 'gpu_private':
  suffix += "_privThr_" + os.environ.get("TF_GPU_THREAD_COUNT", "2")


def run_loop(progressbar):
    loss_history=[]
    update_pbar=hvd.rank()==0
    for i in progressbar:
      features,labels=next(train_iter)
      total_loss=train_step(features,labels,params,mask_rcnn,optimizer)
      loss_history.append(total_loss.numpy())
      smoothed_loss = mean(loss_history[-50:])
      if update_pbar:
        progressbar.set_description(
            "L: {0:.4f},  LR: {1:.4f}".format(
                smoothed_loss,  schedule(optimizer.iterations)))

def do_step_profile(profile_path, stepstr, progressbar):
  if do_profile:
    loss_history=[]
    tf_profiler.start(profile_path)
    print(f"Saving profile to {profile_path}")
    for i in progressbar:
      with prof_Trace(f"{stepstr}_train", step_num=i, _r=1):
        features,labels=next(train_iter)
        total_loss=train_step(features,labels,params,mask_rcnn,optimizer)
      loss_history.append(total_loss.numpy())
      smoothed_loss = mean(loss_history[-50:])
      progressbar.set_description(
          "L: {0:.4f},  LR: {1:.4f}".format(
              smoothed_loss,  schedule(optimizer.iterations)))
    tf_profiler.stop()
  else:
    run_loop(progressbar)

profile_base = "/work/sami/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/Profiles"
if "TFLocal" in tf.__file__:
  profile_path = os.path.join(profile_base,
                              f"Ampere_2.4_tf2_model{suffix}")
stepstr = "local"
steps = 20

if hvd.rank() == 0:
  p_bar = tqdm(range(steps))
else:
  p_bar = range(steps)
run_loop(p_bar)
steps=4000
p_bar=tqdm(range(steps))
loss_history = []
rpn_loss_history = []
rcnn_loss_history = []
timings=[]
#sys.exit()
if do_profile:
  do_step_profile(profile_path,stepstr,p_bar)
else:
  for i in p_bar:
    tstart = time.perf_counter()
    features,labels=next(train_iter)
    total_loss=train_step(features,labels,params,mask_rcnn,optimizer)
    delta_t = time.perf_counter() - tstart
    timings.append(delta_t)
    loss_history.append(total_loss.numpy())
    smoothed_loss = mean(loss_history[-50:])
    if hvd.rank() == 0:
      p_bar.set_description(
          "L: {0:.4f},  LR: {1:.4f}".format(
              smoothed_loss,  schedule(optimizer.iterations)))
  timings = np.asarray(timings, np.float)
  print(f"average step time={np.mean(timings)} +/- {np.std(timings)}")
