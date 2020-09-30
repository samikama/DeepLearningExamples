import os
import sys
from statistics import mean
import copy
import threading
from math import ceil

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from mask_rcnn.utils.logging_formatter import logging
from mask_rcnn.utils.distributed_utils import MPI_is_distributed, MPI_rank, MPI_size, MPI_local_rank
from mpi4py import MPI
import horovod.tensorflow as hvd
hvd.init()

from mask_rcnn import dataloader
from mask_rcnn import distributed_executer
from mask_rcnn.tf2.mask_rcnn_model import SessionModel
from mask_rcnn.hooks import pretrained_restore_hook
from mask_rcnn import evaluation

from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn.hyperparameters import params_io
from mask_rcnn.training import losses, learning_rates
from mask_rcnn import coco_metric
from mask_rcnn.utils import coco_utils

os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ADJUST_HUE_FUSED'] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([devices[MPI_local_rank()]], 'GPU')
logical_devices = tf.config.list_logical_devices('GPU')

def train_epoch(model, sess, steps):
    if MPI_rank()==0:
        p_bar = tqdm(range(steps), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        logging.info("Starting training loop")
        loss_history = []
    else:
        p_bar = range(steps)
    for _ in p_bar:
        model_output = sess.run(model.train_step)
        if MPI_rank()==0:
            loss_history.append(model_output['total_loss'])
            lr = model_output['learning_rate']
            p_bar.set_description("Loss: {0:.4f}, LR: {1:.4f}".format(mean(loss_history[-50:]), lr))
            
def run_eval(model, sess, steps, params, async_eval=False, use_ext=False):
    if MPI_rank()==0:
        p_bar = tqdm(range(steps), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        logging.info("Starting eval loop")
    else:
        p_bar = range(steps)
    worker_predictions = dict()
    for i in p_bar:
        out = sess.run(model.eval_step)
        out = evaluation.process_prediction_for_eval(out)
        for k, v in out.items():
            if k not in worker_predictions:
                worker_predictions[k] = [v]
            else:
                worker_predictions[k].append(v)
    coco = coco_metric.MaskCOCO()
    _preds = copy.deepcopy(worker_predictions)
    for k, v in _preds.items():
        _preds[k] = np.concatenate(v, axis=0)
    if MPI_rank() < 32:
        converted_predictions = coco.load_predictions(_preds, include_mask=True, is_image_mask=False)
        worker_source_ids = _preds['source_id']
    else:
        converted_predictions = []
        worker_source_ids = []
    MPI.COMM_WORLD.barrier()
    predictions_list = evaluation.gather_result_from_all_processes(converted_predictions)
    source_ids_list = evaluation.gather_result_from_all_processes(worker_source_ids)
    validation_json_file=params.val_json_file
    if MPI_rank() == 0:
        all_predictions = []
        source_ids = []
        for i, p in enumerate(predictions_list):
            if i < 32:
                all_predictions.extend(p)
        for i, s in enumerate(source_ids_list):
            if i < 32:
                source_ids.extend(s)
        if use_ext:
            args = [all_predictions, validation_json_file]
            if async_eval:
                eval_thread = threading.Thread(target=evaluation.fast_eval,
                                                   name="eval-thread", args=args)
                eval_thread.start()
            else:
                evaluation.fast_eval(*args)
        else:
            args = [all_predictions, source_ids, True, validation_json_file]
            if async_eval:
                eval_thread = threading.Thread(target=evaluation.compute_coco_eval_metric_n, name="eval-thread", args=args)
                eval_thread.start()
            else:
                evaluation.compute_coco_eval_metric_n(*args)

def train_and_eval(run_config, train_input_fn, eval_input_fn):
    total_epochs = ceil(run_config.total_steps/run_config.num_steps_per_eval)
    model = SessionModel(run_config, train_input_fn, eval_input_fn)
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    var_map = pretrained_restore_hook.build_assigment_map('mrcnn/resnet50/')
    assign_op, feed_dict = pretrained_restore_hook.assign_from_checkpoint(run_config.checkpoint, var_map)
    if MPI_rank()==0:
        hooks.extend([tf.compat.v1.train.CheckpointSaverHook(run_config.model_dir,
                                        save_steps=run_config.total_steps)])
    sess_config = model.get_session_config(use_xla=run_config.xla)
    session_creator=tf.compat.v1.train.ChiefSessionCreator(config=sess_config)
    sess = tf.compat.v1.train.MonitoredSession(session_creator=session_creator, hooks=hooks)
    sess.run(model.train_tdf.initializer)
    sess.run(assign_op, feed_dict=feed_dict)
    eval_workers = min(MPI_size(), 32)
    for epoch in range(run_config.first_eval):
        if MPI_rank()==0:
            logging.info("Starting epoch {} of {}".format(epoch+1, total_epochs))
        train_epoch(model, sess, run_config.num_steps_per_eval)
    for epoch in range(run_config.first_eval, total_epochs):
        if MPI_rank()==0:
            logging.info("Starting epoch {} of {}".format(epoch+1, total_epochs))
        train_epoch(model, sess, run_config.num_steps_per_eval)
        if MPI_rank()==0:
            logging.info("Running epoch {} evaluation".format(epoch+1))
        sess.run(model.eval_tdf.initializer)
        run_eval(model, sess, run_config.eval_samples//eval_workers, run_config, 
                 async_eval=run_config.async_eval, use_ext=run_config.use_ext)
