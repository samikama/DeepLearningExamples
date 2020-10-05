import os
import sys
from math import ceil
import time
os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ADJUST_HUE_FUSED'] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

from mask_rcnn.utils.logging_formatter import logging
from mask_rcnn.utils.distributed_utils import MPI_is_distributed, MPI_rank, MPI_size, MPI_local_rank
from mask_rcnn.tf2.mask_rcnn_model import TapeModel
from mask_rcnn.utils.herring_env import is_herring
import tensorflow as tf


def train_and_eval(run_config, train_input_fn, eval_input_fn):
    
    if is_herring():
        import herring.tensorflow as herring
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus[herring.local_rank()], 'GPU')
    else:
        if MPI_is_distributed(False):
            import horovod.tensorflow as hvd
            hvd.init()
            
        devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices([devices[MPI_local_rank()]], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')

    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": run_config.amp})
    tf.config.optimizer.set_jit(run_config.xla)
    total_epochs = ceil(run_config.total_steps/run_config.num_steps_per_eval)
    mrcnn_model = TapeModel(run_config, train_input_fn, eval_input_fn)
    mrcnn_model.initialize_model()
    eval_workers = min(MPI_size(is_herring()), 32)
    
    if run_config.offload_eval:
        for epoch in range(run_config.first_eval, total_epochs):
            if MPI_rank(is_herring())==0:
                logging.info("Starting epoch {} of {}".format(epoch+1, total_epochs))
            mrcnn_model.train_epoch(run_config.num_steps_per_eval, broadcast=epoch==0)
    
    else:
        for epoch in range(run_config.first_eval):
            if MPI_rank(is_herring())==0:
                logging.info("Starting epoch {} of {}".format(epoch+1, total_epochs))
            mrcnn_model.train_epoch(run_config.num_steps_per_eval, broadcast=epoch==0)
        for epoch in range(run_config.first_eval, total_epochs):
            if MPI_rank(is_herring())==0:
                logging.info("Starting epoch {} of {}".format(epoch+1, total_epochs))
            mrcnn_model.train_epoch(run_config.num_steps_per_eval, broadcast=epoch==0)
            if MPI_rank(is_herring())==0:
                logging.info("Running epoch {} evaluation".format(epoch+1))
            mrcnn_model.run_eval(run_config.eval_samples//eval_workers, async_eval=run_config.async_eval, 
                                 use_ext=run_config.use_ext)
