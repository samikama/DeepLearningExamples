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

import tensorflow as tf

if MPI_is_distributed:
    import horovod.tensorflow as hvd
    hvd.init()
    
devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([devices[MPI_local_rank()]], 'GPU')
logical_devices = tf.config.list_logical_devices('GPU')

def train_and_eval(run_config, train_input_fn, eval_input_fn):
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": run_config.amp})
    tf.config.optimizer.set_jit(run_config.xla)
    total_epochs = ceil(run_config.total_steps/run_config.num_steps_per_eval)
    mrcnn_model = TapeModel(run_config, train_input_fn, eval_input_fn)
    mrcnn_model.initialize_model()
    eval_workers = min(MPI_size(), 32)
    # eval_workers = MPI_size()
    start_time = time.time()
    for epoch in range(run_config.first_eval):
        if MPI_rank()==0:
            logging.info("Starting epoch {} of {}".format(epoch+1, total_epochs))
            mpi_size = MPI_size()
            logging.info("Number of GPUs {}".format(mpi_size))
        mrcnn_model.train_epoch(run_config.num_steps_per_eval)
    for epoch in range(run_config.first_eval, total_epochs):
        if MPI_rank()==0:
            logging.info("Starting epoch {} of {}".format(epoch+1, total_epochs))
        mrcnn_model.train_epoch(run_config.num_steps_per_eval)
        if MPI_rank()==0:
            logging.info("Running epoch {} evaluation on {} workers".format(epoch+1, eval_workers))
        mrcnn_model.run_eval(run_config.eval_samples//eval_workers+1, async_eval=run_config.async_eval, 
                             use_ext=run_config.use_ext)
    if MPI_rank()==0:
        logging.info("Training complete in {} seconds.".format(time.time() - start_time))

