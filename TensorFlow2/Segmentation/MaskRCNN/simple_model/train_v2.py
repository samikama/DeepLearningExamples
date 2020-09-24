import os
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
from tensorflow.python.profiler import profiler_v2 as tf_profiler
from tensorflow.python.profiler.trace import Trace as prof_Trace

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
tf.disable_v2_behavior()
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

import horovod.tensorflow as hvd
hvd.init()

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[hvd.rank()], True)
tf.config.set_visible_devices(physical_devices[hvd.rank()], 'GPU')
devices = tf.config.list_logical_devices('GPU')

from mask_rcnn.hyperparameters import dataset_params
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn import dataset_utils
from mask_rcnn import dataloader
from mask_rcnn import mask_rcnn_model
import load_weights, model_v2

train_file_pattern = '/home/ubuntu/data/coco/tf_record/train*'
orig_file_pattern = '/data/coco/coco-2017/tfrecord/train*'
batch_size = 1
data_params = dataset_params.get_data_params()
params = mask_rcnn_params.default_config().values()
data_params['batch_size'] = batch_size
params['finetune_bn'] = False
params['train_batch_size'] = batch_size
params['l2_weight_decay'] = 1e-4
params['init_learning_rate'] = 4e-3 * batch_size * hvd.size()
params['warmup_learning_rate'] = 4e-4 * batch_size * hvd.size()
params['warmup_steps'] = 1000
params['learning_rate_steps'] = [30000, 40000]
params['learning_rate_levels'] = [
    4e-4 * batch_size * hvd.size(), 4e-5 * batch_size * hvd.size()
]
params['momentum'] = 0.9
params['use_batched_nms'] = False
params['use_custom_box_proposals_op'] = True
params['amp'] = True
params['include_groundtruth_in_features'] = True
params['use_default_roi_align'] = False
params['use_transposed_features'] = True

orig_loader = dataloader.InputReader(orig_file_pattern, use_instance_mask=True)
orig_tdf = orig_loader(data_params)
orig_iter = tf.data.make_initializable_iterator(orig_tdf)
orig_features, orig_labels = orig_iter.get_next()

mask_rcnn = model_v2.MRCNN(params)

train_outputs = model_v2.model_fn(orig_features,
                                  orig_labels,
                                  params,
                                  mask_rcnn,
                                  is_training=True)

# predictions = model_v2.model_fn(orig_features,
#                                 orig_labels,
#                                 params,
#                                 mask_rcnn,
#                                 is_training=False)

var_list = load_weights.build_assigment_map('mrcnn/resnet50/')
checkpoint_file = tf.train.latest_checkpoint(
    '../weights/resnet/resnet-nhwc-2018-02-07/')
_init_op, _init_feed_dict = load_weights.assign_from_checkpoint(
    checkpoint_file, var_list)
xlaflags = os.environ.get("TF_XLA_FLAGS", "")


def get_env_value(env_str, key):
  pos = env_str.find(key)
  if pos >= 0:
    substr = env_str[pos + len(key)]
    if substr.find(" ") == -1:
      return substr
    else:
      return substr[:substr.find(" ")]
  return None

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

  # pos = xlaflags.find("min_cluster_size=")
  # if pos >= 0:
  #   substr = xlaflags[pos + len("min_cluster_size=")]
  #   if substr.find(" ") == -1:
  #     suffix += "_csize" + substr
  #   else:
  #     suffix += "_csize" + substr[:substr.find(" ")]

if os.environ.get("TF_GPU_THREAD_MODE") == 'gpu_private':
  suffix += "_privThr_" + os.environ.get("TF_GPU_THREAD_COUNT", "2")


def do_step_profile(profile_path, sess, stepstr, progressbar, fetch_ops):
  if do_profile:
    tf_profiler.start(profile_path)
    print(f"Saving profile to {profile_path}")
    for i in progressbar:
      with prof_Trace(f"{stepstr}_train", step_num=i, _r=1):
        outputs = sess.run(fetch_ops)
      loss_history.append(outputs[1])
      rpn_loss_history.append(outputs[2])
      rcnn_loss_history.append(outputs[4])
      smoothed_loss = mean(loss_history[-50:])
      smoothed_rpn_loss = mean(rpn_loss_history[-50:])
      smoothed_rcnn_loss = mean(rcnn_loss_history[-50:])
      progressbar.set_description(
          "L: {0:.4f}, R: {1:.4f}, C: {2:.4f}, LR: {3:.4f}".format(
              smoothed_loss, smoothed_rpn_loss, smoothed_rcnn_loss, outputs[6]))
    tf_profiler.stop()
  else:
    for i in progressbar:
      #       with tf.profiler.experimental.Trace(f"{stepstr}_train",step_num=i,_r=1):
      outputs = sess.run(fetch_ops)
      loss_history.append(outputs[1])
      rpn_loss_history.append(outputs[2])
      rcnn_loss_history.append(outputs[4])
      smoothed_loss = mean(loss_history[-50:])
      smoothed_rpn_loss = mean(rpn_loss_history[-50:])
      smoothed_rcnn_loss = mean(rcnn_loss_history[-50:])
      progressbar.set_description(
          "L: {0:.4f}, R: {1:.4f}, C: {2:.4f}, LR: {3:.4f}".format(
              smoothed_loss, smoothed_rpn_loss, smoothed_rcnn_loss, outputs[6]))


sess = tf.Session()
sess.run(orig_iter.initializer)
sess.run(tf.global_variables_initializer())
sess.run(_init_op, _init_feed_dict)

outputs = sess.run(train_outputs)
# pred = sess.run(predictions)
profile_base = "/work/sami/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/Profiles"
if "TFLocal" in tf.__file__:
  profile_path = os.path.join(profile_base,
                              f"Ampere_2.4_tf1_model{suffix}")
stepstr = "local"
steps = 20

if hvd.rank() == 0 and False:
  p_bar = tqdm(range(steps))
else:
  p_bar = range(steps)
loss_history = []
rpn_loss_history = []
rcnn_loss_history = []
for i in p_bar:
  outputs = sess.run(train_outputs)
  loss_history.append(outputs[1])
  rpn_loss_history.append(outputs[2])
  rcnn_loss_history.append(outputs[4])
  smoothed_loss = mean(loss_history[-50:])
  smoothed_rpn_loss = mean(rpn_loss_history[-50:])
  smoothed_rcnn_loss = mean(rcnn_loss_history[-50:])
  # if hvd.rank() == 0:
  #   p_bar.set_description(
  #       "L: {0:.4f}, R: {1:.4f}, C: {2:.4f}, LR: {3:.4f}".format(
  #           smoothed_loss, smoothed_rpn_loss, smoothed_rcnn_loss, outputs[6]))
steps=4000
p_bar=tqdm(range(steps))
loss_history = []
rpn_loss_history = []
rcnn_loss_history = []
timings=[]
#sys.exit()
if do_profile:
  do_step_profile(profile_path,sess,stepstr,p_bar,train_outputs)
else:
  for i in p_bar:
    tstart = time.perf_counter()
    outputs = sess.run(train_outputs)
    delta_t = time.perf_counter() - tstart
    timings.append(delta_t)
    loss_history.append(outputs[1])
    rpn_loss_history.append(outputs[2])
    rcnn_loss_history.append(outputs[4])
    smoothed_loss = mean(loss_history[-50:])
    smoothed_rpn_loss = mean(rpn_loss_history[-50:])
    smoothed_rcnn_loss = mean(rcnn_loss_history[-50:])
    if hvd.rank() == 0:
      p_bar.set_description(
          "L: {0:.4f}, R: {1:.4f}, C: {2:.4f}, LR: {3:.4f}".format(
              smoothed_loss, smoothed_rpn_loss, smoothed_rcnn_loss, outputs[6]))
  timings = np.asarray(timings, np.float)
  print(f"average step time={np.mean(timings)} +/- {np.std(timings)}")
