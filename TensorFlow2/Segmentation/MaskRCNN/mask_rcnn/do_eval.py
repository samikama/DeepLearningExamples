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

import time

def run_eval(model, sess, steps, params, async_eval=False):
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
        args = [all_predictions, source_ids, True, validation_json_file]
        if async_eval:
            eval_thread = threading.Thread(target=evaluation.compute_coco_eval_metric_n, name="eval-thread", args=args)
            eval_thread.start()
        else:
            evaluation.compute_coco_eval_metric_n(*args)

def do_eval(run_config, train_input_fn, eval_input_fn):
    model = SessionModel(run_config, train_input_fn, eval_input_fn)
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    out_path = '/home/ubuntu/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/results_session_1x/'
    last = out_path+'model.ckpt-0'
    saver = tf.compat.v1.train.Saver()
    sess_config = model.get_session_config(use_xla=run_config.xla)
    session_creator=tf.compat.v1.train.ChiefSessionCreator(config=sess_config)
    sess = tf.compat.v1.train.MonitoredSession(session_creator=session_creator, hooks=hooks)
    
    eval_workers = min(32, MPI_size()) 
    latest = tf.train.latest_checkpoint("/home/ubuntu/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/results_session_1x/")
    while True:
        if last != latest:
            print("#"*20, "New latest found", latest)
            last = latest
            sess.run(model.eval_tdf.initializer)
            saver.restore(sess,latest)
            run_eval(model, sess, run_config.eval_samples//eval_workers, run_config) 
        else:
            latest = tf.train.latest_checkpoint("/home/ubuntu/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/results_session_1x/")
            print("#"*20,"Nothing new here")
            time.sleep(5)

