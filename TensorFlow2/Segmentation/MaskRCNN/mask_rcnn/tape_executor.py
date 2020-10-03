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
from mask_rcnn.tf2.mask_rcnn_model import TapeModel
from mask_rcnn.utils.herring_env import is_herring

if is_herring():

    from mask_rcnn.utils.distributed_utils_herring import MPI_is_distributed, MPI_rank, MPI_size, MPI_local_rank
else:
    from mask_rcnn.utils.distributed_utils import MPI_is_distributed, MPI_rank, MPI_size, MPI_local_rank

import tensorflow as tf

if MPI_is_distributed:
    import horovod.tensorflow as hvd
    hvd.init()
    
#devices = tf.config.list_physical_devices('GPU')
#tf.config.set_visible_devices(list(range(len(os.environ.get("CUDA_VISIBLE_DEVICES","0").split(",")))),"GPU")
#logical_devices = tf.config.list_logical_devices('GPU')

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
            
        # devices = tf.config.list_physical_devices('GPU')
        # tf.config.set_visible_devices([devices[MPI_local_rank()]], 'GPU')
        #logical_devices = tf.config.list_logical_devices('GPU')

    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": run_config.amp})
    tf.config.optimizer.set_jit(run_config.xla)
    total_epochs = ceil(run_config.total_steps/run_config.num_steps_per_eval)
    mrcnn_model = TapeModel(run_config, train_input_fn, eval_input_fn)
    mrcnn_model.initialize_model()
    eval_workers = min(MPI_size(is_herring()), 32) if run_config.dist_eval else 1
    start_time = time.time()
    eval_each_step=False
    if run_config.mode.lower() == "train_and_eval":
        eval_each_step=True
    profile_path=None
    if run_config.profile_path:
        profile_path=run_config.profile_path
    logging.info("SAMI SAMI SAMI Profile path is {}".format(profile_path))
    
    if run_config.offload_eval:
        for epoch in range(run_config.first_eval, total_epochs):
            if MPI_rank(is_herring())==0:
                logging.info("Starting epoch {} of {}".format(epoch+1, total_epochs))
                mpi_size = MPI_size()
                logging.info("Number of GPUs {}".format(mpi_size))
            mrcnn_model.train_epoch(min(run_config.num_steps_per_eval,run_config.total_steps), broadcast=epoch==0,profile=f"{profile_path}_{epoch}" if profile_path else None)
    
    else:
        for epoch in range(run_config.first_eval):
            if MPI_rank(is_herring())==0:
                logging.info("Starting epoch {} of {}".format(epoch+1, total_epochs))
            mrcnn_model.train_epoch(min(run_config.num_steps_per_eval,run_config.total_steps), broadcast=epoch==0,profile=f"{profile_path}_{epoch}" if profile_path else None)
        for epoch in range(run_config.first_eval, total_epochs):
            if MPI_rank(is_herring())==0:
                logging.info("Starting epoch {} of {}".format(epoch+1, total_epochs))
            mrcnn_model.train_epoch(min(run_config.num_steps_per_eval,run_config.total_steps), broadcast=epoch==0,profile=f"{profile_path}_{epoch}" if profile_path else None)
            if MPI_rank(is_herring())==0 and eval_each_step:
                logging.info("Running epoch {} evaluation".format(epoch+1))
            if eval_each_step:
                mrcnn_model.run_eval(run_config.eval_samples//eval_workers, async_eval=run_config.async_eval, 
                                    use_ext=run_config.use_ext)
    if run_config.eval_after_training:
        if run_config.dist_eval:
            mrcnn_model.run_eval(run_config.eval_samples//eval_workers, async_eval=run_config.async_eval, 
                                        use_ext=run_config.use_ext)
        else:
            if MPI_rank(is_herring())==0:
                mrcnn_model.run_eval(run_config.eval_samples//eval_workers, async_eval=run_config.async_eval, 
                                        use_ext=run_config.use_ext)
    if MPI_rank(is_herring())==0:
        logging.info("Training complete in {} seconds.".format(time.time() - start_time))

    if MPI_rank()==0:
        logging.info("Training complete in {} seconds.".format(time.time() - start_time))

def eval(run_config, train_input_fn, eval_input_fn):
    if is_herring():
        import herring.tensorflow as herring
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus[herring.local_rank()], 'GPU')
    else:
        if MPI_is_distributed(False):
            import horovod.tensorflow as hvd
            hvd.init()
            
        # devices = tf.config.list_physical_devices('GPU')
        # tf.config.set_visible_devices([devices[MPI_local_rank()]], 'GPU')
        #logical_devices = tf.config.list_logical_devices('GPU')

    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": run_config.amp})
    tf.config.optimizer.set_jit(run_config.xla)
    total_epochs = ceil(run_config.total_steps/run_config.num_steps_per_eval)
    mrcnn_model = TapeModel(run_config, train_input_fn, eval_input_fn,is_training=False)
    mrcnn_model.initialize_model()
    last_checkpoint=sorted([os.path.join(run_config.model_dir,x) for x in os.listdir(run_config.model_dir)])[-1]
    mrcnn_model.load_model(last_checkpoint)
    eval_workers = min(MPI_size(is_herring()), 32) if run_config.dist_eval else 1
    start_time = time.time()
    eval_each_step=False
    tstart=time.perf_counter()
    if run_config.dist_eval:
        mrcnn_model.run_eval(run_config.eval_samples//eval_workers, async_eval=run_config.async_eval, 
                                    use_ext=run_config.use_ext)
    else:
        if MPI_rank(is_herring())==0:
            mrcnn_model.run_eval(run_config.eval_samples//eval_workers, async_eval=run_config.async_eval, 
                                    use_ext=run_config.use_ext)
    tend=time.perf_counter()
    if MPI_rank(is_herring())==0:
        logging.info("Eval complete in {} seconds.".format(tend - tstart))
