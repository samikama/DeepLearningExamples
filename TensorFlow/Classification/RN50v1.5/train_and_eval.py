import os

import warnings
warnings.simplefilter("ignore")

import tensorflow as tf

import horovod.tensorflow as hvd
from utils import hvd_utils

from runtime import Runner

# mpiexec --allow-run-as-root --bind-to socket -np 8 python train_and_eval.py

if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)
    
    RUNNING_CONFIG = tf.contrib.training.HParams(
        mode='train',

        # ======= Directory HParams ======= #
        log_dir='/home/ubuntu/results',
        model_dir='/home/ubuntu/model',
        summaries_dir='/home/ubuntu/summaries',
        data_dir='/home/ubuntu/data/tf-imagenet',
        data_idx_dir=None,

        # ========= Model HParams ========= #
        n_classes=1001,
        input_format='NHWC',
        compute_format='NHWC',
        dtype=tf.float16,
        height=224,
        width=224,
        n_channels=3,

        # ======= Training HParams ======== #
        iter_unit='epoch',
        num_iter=1,
        warmup_steps=50,
        batch_size=256,
        log_every_n_steps=100,
        learning_rate_init=0.01,
        weight_decay=1e-4,
        momentum=0.9,
        loss_scale=256,
        use_static_loss_scaling=False,
        distort_colors=False,

        # ======= Optimization HParams ======== #
        use_xla=True,
        use_tf_amp=False,
        use_dali=False,
        gpu_memory_fraction=1,
        
        seed=None,
    )
    
    runner = Runner(
        # ========= Model HParams ========= #
        n_classes=RUNNING_CONFIG.n_classes,
        input_format=RUNNING_CONFIG.input_format,
        compute_format=RUNNING_CONFIG.compute_format,
        dtype=RUNNING_CONFIG.dtype,
        n_channels=RUNNING_CONFIG.n_channels,
        height=RUNNING_CONFIG.height,
        width=RUNNING_CONFIG.width,
        distort_colors=RUNNING_CONFIG.distort_colors,
        log_dir=RUNNING_CONFIG.log_dir,
        model_dir=RUNNING_CONFIG.model_dir,
        data_dir=RUNNING_CONFIG.data_dir,
        data_idx_dir=RUNNING_CONFIG.data_idx_dir,

        # ======= Optimization HParams ======== #
        use_xla=RUNNING_CONFIG.use_xla,
        use_tf_amp=RUNNING_CONFIG.use_tf_amp,
        use_dali=RUNNING_CONFIG.use_dali,
        gpu_memory_fraction=RUNNING_CONFIG.gpu_memory_fraction,
        seed=RUNNING_CONFIG.seed
    )
    
    for i in range(10):
        runner.train(
                    iter_unit=RUNNING_CONFIG.iter_unit,
                    num_iter=RUNNING_CONFIG.num_iter,
                    batch_size=RUNNING_CONFIG.batch_size,
                    warmup_steps=RUNNING_CONFIG.warmup_steps,
                    log_every_n_steps=RUNNING_CONFIG.log_every_n_steps,
                    weight_decay=RUNNING_CONFIG.weight_decay,
                    learning_rate_init=RUNNING_CONFIG.learning_rate_init,
                    momentum=RUNNING_CONFIG.momentum,
                    loss_scale=RUNNING_CONFIG.loss_scale,
                    use_static_loss_scaling=False,
                    is_benchmark=RUNNING_CONFIG.mode == 'training_benchmark',
                )

        print("\n\n epoch: {}\n\n".format(i))

        runner.evaluate(
                    iter_unit='batch',
                    num_iter=10,
                    warmup_steps=RUNNING_CONFIG.warmup_steps,
                    batch_size=RUNNING_CONFIG.batch_size,
                    log_every_n_steps=RUNNING_CONFIG.log_every_n_steps,
                    is_benchmark=False
                )
        