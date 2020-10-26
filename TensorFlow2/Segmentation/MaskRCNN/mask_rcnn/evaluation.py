#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions to perform COCO evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import operator
import pprint
import six
import time
import itertools
import collections
import io
import threading
import time
import json

from PIL import Image

import numpy as np
import tensorflow as tf

from mask_rcnn.utils.logging_formatter import logging

from mask_rcnn import coco_metric
from mask_rcnn.utils import coco_utils

from mask_rcnn.object_detection import visualization_utils
from mask_rcnn.utils.distributed_utils import MPI_rank

import dllogger
from dllogger import Verbosity

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import multiprocessing as mp
import cProfile, pstats

#mp.set_start_method('spawn')

def profile_dec(func):
  def wrapper(*args, **kwargs):
    with cProfile.Profile() as pr:
      ret = func(*args, **kwargs)
      ps = pstats.Stats(pr).sort_stats('cumtime')
      ps.print_stats()
    return ret
  
  return wrapper

def process_prediction_for_eval(prediction):
    """Process the model prediction for COCO eval."""
    print("Still running the normal Eval")
    image_info = prediction['image_info']
    box_coordinates = prediction['detection_boxes']
    processed_box_coordinates = np.zeros_like(box_coordinates)

    for image_id in range(box_coordinates.shape[0]):
        scale = image_info[image_id][2]

        for box_id in range(box_coordinates.shape[1]):
            # Map [y1, x1, y2, x2] -> [x1, y1, w, h] and multiply detections
            # Map [y1, x1, y2, x2] -> [x1, y1, w, h] and multiply detections
            # by image scale.
            y1, x1, y2, x2 = box_coordinates[image_id, box_id, :]
            new_box = scale * np.array([x1, y1, x2 - x1, y2 - y1])
            processed_box_coordinates[image_id, box_id, :] = new_box
    prediction['detection_boxes'] = processed_box_coordinates
    return prediction

def process_prediction_for_eval_batch(prediction):

    image_info = prediction['image_info']
    box_coordinates = prediction['detection_boxes']
    processed_box_coordinates = np.zeros_like(box_coordinates)
    
    for image_id in range(box_coordinates.shape[0]):
        scale = image_info[image_id][2]
        processed_box_coordinates[image_id, :] = np.concatenate((box_coordinates[image_id, :, 1][..., np.newaxis], box_coordinates[image_id, :, 0][..., np.newaxis], 
                        box_coordinates[image_id, :, 3][..., np.newaxis] - box_coordinates[image_id, :, 1][..., np.newaxis], 
                        box_coordinates[image_id, :, 2][..., np.newaxis] - box_coordinates[image_id, :, 0][..., np.newaxis]), axis=1) * scale
    prediction['detection_boxes'] = processed_box_coordinates
    return prediction


def get_predictions(predictor,
                    num_batches=-1,
                    eval_batch_size=-1):

    predictions = dict()
    batch_idx = 0

    T0 = time.time()
    while num_batches < 0 or batch_idx < num_batches:

        try:
            step_t0 = time.time()
            step_predictions = six.next(predictor)
            batch_time = time.time() - step_t0

            throughput = eval_batch_size / batch_time

            logging.info('Running inference on batch %03d/%03d... - Step Time: %.4fs - Throughput: %.1f imgs/s' % (
                batch_idx + 1,
                num_batches,
                batch_time,
                throughput
            ))

        except StopIteration:
            logging.info('Get StopIteration at %d batch.' % (batch_idx + 1))
            break

        step_predictions = process_prediction_for_eval(step_predictions)

        for k, v in step_predictions.items():

            if k not in predictions:
                predictions[k] = [v]

            else:
                predictions[k].append(v)

        batch_idx = batch_idx + 1
    logging.info('Inference took {}s'.format(time.time() - T0))
    return predictions


def compute_coco_eval_metric_n(predictions,
                             source_ids,
                             include_mask=True,
                             annotation_json_file=""):
    """Compute COCO eval metric given a predictions.
    """
    logging.info("Metric calculation started {}".format(time.time()))
    if annotation_json_file == "":
        annotation_json_file = None

    eval_metric = coco_metric.EvaluationMetric(annotation_json_file, include_mask=include_mask)
    eval_results = eval_metric.predict_metric_fn2(predictions, source_ids)

    logging.info("==================== Metrics ====================")

    # logging.info('Eval Epoch results: %s' % pprint.pformat(eval_results, indent=4))
    for key, value in sorted(eval_results.items(), key=operator.itemgetter(0)):
        logging.info("%s: %.9f" % (key, value))
    print()  # Visual Spacing
    logging.info("Metric calculation ended {}".format(time.time()))
    return eval_results





def compute_coco_eval_metric_1(predictor,
                             num_batches=-1,
                             include_mask=True,
                             annotation_json_file="",
                             eval_batch_size=-1,
                             report_frequency=None):
    """Compute COCO eval metric given a prediction generator.

    Args:
    predictor: a generator that iteratively pops a dictionary of predictions
      with the format compatible with COCO eval tool.
    num_batches: the number of batches to be aggregated in eval. This is how
      many times that the predictor gets pulled.
    include_mask: a boolean that indicates whether we include the mask eval.
    annotation_json_file: the annotation json file of the eval dataset.

    Returns:
    eval_results: the aggregated COCO metric eval results.
    """

    if annotation_json_file == "":
        annotation_json_file = None

    use_groundtruth_from_json = (annotation_json_file is not None)

    predictions = dict()
    batch_idx = 0

    if use_groundtruth_from_json:
        eval_metric = coco_metric.EvaluationMetric(annotation_json_file, include_mask=include_mask)
    else:
        eval_metric = coco_metric.EvaluationMetric(filename=None, include_mask=include_mask)

    def evaluation_preds(preds):

        # Essential to avoid modifying the source dict
        _preds = copy.deepcopy(preds)

        for k, v in six.iteritems(_preds):
            _preds[k] = np.concatenate(_preds[k], axis=0)

        if 'orig_images' in _preds and _preds['orig_images'].shape[0] > 10:
            # Only samples a few images for visualization.
            _preds['orig_images'] = _preds['orig_images'][:10]

        if use_groundtruth_from_json:
            eval_results = eval_metric.predict_metric_fn(_preds)

        else:
            images, annotations = coco_utils.extract_coco_groundtruth(_preds, include_mask)
            coco_dataset = coco_utils.create_coco_format_dataset(images, annotations)
            eval_results = eval_metric.predict_metric_fn(_preds, groundtruth_data=coco_dataset)

        return eval_results

    # Take into account cuDNN & Tensorflow warmup
    # Drop N first steps for avg throughput calculation
    BURNIN_STEPS = 100
    model_throughput_list = list()
    inference_time_list = list()

    while num_batches < 0 or batch_idx < num_batches:

        try:
            step_t0 = time.time()
            step_predictions = six.next(predictor)
            batch_time = time.time() - step_t0

            throughput = eval_batch_size / batch_time
            model_throughput_list.append(throughput)
            inference_time_list.append(batch_time)

            logging.info('Running inference on batch %03d/%03d... - Step Time: %.4fs - Throughput: %.1f imgs/s' % (
                batch_idx + 1,
                num_batches,
                batch_time,
                throughput
            ))

        except StopIteration:
            logging.info('Get StopIteration at %d batch.' % (batch_idx + 1))
            break

        step_predictions = process_prediction_for_eval(step_predictions)

        for k, v in step_predictions.items():

            if k not in predictions:
                predictions[k] = [v]

            else:
                predictions[k].append(v)

        batch_idx = batch_idx + 1

        # If you want the report to happen each report_frequency to happen each report_frequency batches.
        # Thus, each report is of eval_batch_size * report_frequency
        if report_frequency and batch_idx % report_frequency == 0:
            eval_results = evaluation_preds(preds=predictions)
            logging.info('Eval results: %s' % pprint.pformat(eval_results, indent=4))

    inference_time_list.sort()
    logging.info(predictions['source_id'])
    eval_results = evaluation_preds(preds=predictions)

    average_time = np.mean(inference_time_list)
    latency_50 = max(inference_time_list[:int(len(inference_time_list) * 0.5)])
    latency_90 = max(inference_time_list[:int(len(inference_time_list) * 0.90)])
    latency_95 = max(inference_time_list[:int(len(inference_time_list) * 0.95)])
    latency_99 = max(inference_time_list[:int(len(inference_time_list) * 0.99)])
    latency_100 = max(inference_time_list[:int(len(inference_time_list) * 1)])

    print()  # Visual Spacing
    logging.info("# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #")
    logging.info("         Evaluation Performance Summary          ")
    logging.info("# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #")

    total_processing_hours, rem = divmod(np.sum(model_throughput_list), 3600)
    total_processing_minutes, total_processing_seconds = divmod(rem, 60)

    if len(model_throughput_list) > BURNIN_STEPS:
        # Take into account cuDNN & Tensorflow warmup
        # Drop N first steps for avg throughput calculation
        # Also drop last step which may have a different batch size
        avg_throughput = np.mean(model_throughput_list[BURNIN_STEPS:-1])
    else:
        avg_throughput = -1.

    print()  # Visual Spacing
    logging.info("Average throughput: {throughput:.1f} samples/sec".format(throughput=avg_throughput))
    logging.info("Inference Latency Average (s) = {avg:.4f}".format(avg=average_time))
    logging.info("Inference Latency 50% (s) = {cf_50:.4f}".format(cf_50=latency_50))
    logging.info("Inference Latency 90%  (s) = {cf_90:.4f}".format(cf_90=latency_90))
    logging.info("Inference Latency 95%  (s) = {cf_95:.4f}".format(cf_95=latency_95))
    logging.info("Inference Latency 99%  (s) = {cf_99:.4f}".format(cf_99=latency_99))
    logging.info("Inference Latency 100%  (s) = {cf_100:.4f}".format(cf_100=latency_100))
    logging.info("Total processed steps: {total_steps}".format(total_steps=len(model_throughput_list)))
    logging.info(
        "Total processing time: {hours}h {minutes:02d}m {seconds:02d}s".format(
            hours=total_processing_hours,
            minutes=int(total_processing_minutes),
            seconds=int(total_processing_seconds)
        )
    )
    dllogger.log(step=(), data={"avg_inference_throughput": avg_throughput}, verbosity=Verbosity.DEFAULT)
    avg_inference_time = float(total_processing_hours * 3600 + int(total_processing_minutes) * 60 +
        int(total_processing_seconds))
    dllogger.log(step=(), data={"avg_inference_time": avg_inference_time}, verbosity=Verbosity.DEFAULT)
    logging.info("==================== Metrics ====================")

    # logging.info('Eval Epoch results: %s' % pprint.pformat(eval_results, indent=4))
    for key, value in sorted(eval_results.items(), key=operator.itemgetter(0)):
        logging.info("%s: %.9f" % (key, value))
    print()  # Visual Spacing
    return eval_results, predictions

def gather_result_from_all_processes(local_results, root=0):
  
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  print(len(local_results), flush=True)
  res = comm.gather(local_results, root=0)
  
  return res

def evaluate(eval_estimator,
             input_fn,
             num_eval_samples,
             eval_batch_size,
             include_mask=True,
             validation_json_file="",
             report_frequency=None,
             latest_ckpt=None,
             do_distributed=False,
             async_eval=False,
             use_ext=False):

    """Runs COCO evaluation once."""
    predictor = eval_estimator.predict(
        input_fn=input_fn,
        checkpoint_path=latest_ckpt,
        yield_single_examples=False
    )

    # Every predictor.next() gets a batch of prediction (a dictionary).
    num_eval_times = num_eval_samples // eval_batch_size
    assert num_eval_times > 0, 'num_eval_samples must be >= eval_batch_size!'

    if do_distributed:
        worker_predictions = get_predictions(predictor,
            num_batches=num_eval_times,
            eval_batch_size=eval_batch_size)
        logging.info(worker_predictions['source_id'])
        # DEBUG - print worker predictions
        # _ = compute_coco_eval_metric_n(worker_predictions, False, validation_json_file)
        coco = coco_metric.MaskCOCO()
        _preds = copy.deepcopy(worker_predictions)
        for k, v in _preds.items():
            # combined all results in flat structure for eval
            _preds[k] = np.concatenate(v, axis=0)
        converted_predictions = coco.load_predictions(_preds, include_mask=True, is_image_mask=False)
        worker_source_ids = _preds['source_id']

        # logging.info(converted_predictions)
        # gather on rank 0
        predictions_list = gather_result_from_all_processes(converted_predictions)
        source_ids_list = gather_result_from_all_processes(worker_source_ids)
        if MPI_rank() == 0:
            all_predictions = []
            source_ids = []
            for i, p in enumerate(predictions_list):
                all_predictions.extend(p)
            for i, s in enumerate(source_ids_list):
                source_ids.extend(s)

            # run metric calculation on root node TODO: launch this in it's own thread
            if use_ext:
                args = [all_predictions, validation_json_file]
                if async_eval:
                    eval_thread = threading.Thread(target=fast_eval,
                                                   name="eval-thread", args=args)
                    eval_thread.start()
                    return eval_thread
                else:
                    fast_eval(*args)
            else:
                args = [all_predictions, source_ids, include_mask, validation_json_file]
                if async_eval:
                    eval_thread = threading.Thread(target=compute_coco_eval_metric_n, name="eval-thread", args=args)
                    eval_thread.start()
                    return eval_thread
                else:
                    compute_coco_eval_metric_n(*args)
            logging.info("Launched compute eval metrics thread ...")
        return None
    else:
        eval_results, predictions = compute_coco_eval_metric_1(
            predictor,
            num_eval_times,
            include_mask,
            validation_json_file,
            eval_batch_size=eval_batch_size,
            report_frequency=report_frequency
        )

        return eval_results, predictions


def write_summary(eval_results, summary_dir, current_step, predictions=None):
    """Write out eval results for the checkpoint."""
    with tf.Graph().as_default():
        summaries = []

        # Summary writer writes out eval metrics.
        try:
            # Tensorflow 1.x
            summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)
        except AttributeError:
            # Tensorflow 2.x
            summary_writer = tf.summary.create_file_writer(summary_dir)
            summary_writer.as_default()

        eval_results_dict = {}
        for metric in eval_results:
            try:
                summaries.append(tf.compat.v1.Summary.Value(tag=metric, simple_value=eval_results[metric]))
                eval_results_dict[metric] = float(eval_results[metric])

            except AttributeError:
                tf.summary.scalar(name=metric, data=eval_results[metric], step=current_step)
                eval_results_dict[metric] = float(eval_results[metric])
        dllogger.log(step=(), data=eval_results_dict, verbosity=Verbosity.DEFAULT)

        if isinstance(predictions, dict) and predictions:
            images_summary = get_image_summary(predictions, current_step)

            try:
                summaries += images_summary
            except TypeError:
                summaries.append(images_summary)

        try:
            # tf_summaries = tf.compat.v1.Summary(value=list(summaries))
            tf_summaries = tf.compat.v1.Summary(value=summaries)
            summary_writer.add_summary(tf_summaries, current_step)
            summary_writer.flush()

        except AttributeError:
            tf.summary.flush(summary_writer)


def generate_image_preview(image, boxes, scores, classes, gt_boxes=None, segmentations=None):
    """Creates an image summary given predictions."""
    max_boxes_to_draw = 100
    min_score_thresh = 0.1

    # Visualizes the predicitons.
    image_with_detections = visualization_utils.visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes=classes,
        scores=scores,
        category_index={},
        instance_masks=segmentations,
        use_normalized_coordinates=False,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False
    )

    if gt_boxes is not None:
        # Visualizes the groundtruth boxes. They are in black by default.
        image_with_detections = visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_with_detections,
            gt_boxes,
            classes=None,
            scores=None,
            category_index={},
            use_normalized_coordinates=False,
            max_boxes_to_draw=max_boxes_to_draw,
            agnostic_mode=True
        )

    return image_with_detections


def generate_image_buffer(input_image):
    buf = io.BytesIO()
    w, h = input_image.shape[:2]
    ratio = 1024 / w
    new_size = [int(w * ratio), int(h * ratio)]

    image = Image.fromarray(input_image.astype(np.uint8))
    image.thumbnail(new_size)
    image.save(buf, format='png')

    return buf.getvalue()


def get_image_summary(predictions, current_step, max_images=10):
    """Write out image and prediction for summary."""

    if 'orig_images' not in predictions:
        logging.info('Missing orig_images in predictions: %s', predictions.keys())
        return

    max_images = min(
        len(predictions['orig_images']) * predictions['orig_images'][0].shape[0],
        max_images
    )

    _detection_boxes = np.concatenate(predictions['detection_boxes'], axis=0)
    _detection_scores = np.concatenate(predictions['detection_scores'], axis=0)
    _detection_classes = np.concatenate(predictions['detection_classes'], axis=0)
    _image_info = np.concatenate(predictions['image_info'], axis=0)
    _num_detections = np.concatenate(predictions['num_detections'], axis=0)
    _orig_images = np.concatenate(predictions['orig_images'], axis=0)

    if 'detection_masks' in predictions:
        _detection_masks = np.concatenate(predictions['detection_masks'], axis=0)
    else:
        _detection_masks = None

    if 'groundtruth_boxes' in predictions:
        _groundtruth_boxes = np.concatenate(predictions['groundtruth_boxes'], axis=0)
    else:
        _groundtruth_boxes = None

    _orig_images = _orig_images * 255
    _orig_images = _orig_images.astype(np.uint8)

    image_previews = []

    for i in range(max_images):
        num_detections = min(len(_detection_boxes[i]), int(_num_detections[i]))

        detection_boxes = _detection_boxes[i][:num_detections]
        detection_scores = _detection_scores[i][:num_detections]
        detection_classes = _detection_classes[i][:num_detections]

        image = _orig_images[i]
        image_height = image.shape[0]
        image_width = image.shape[1]

        # Rescale the box to fit the visualization image.
        h, w = _image_info[i][3:5]
        detection_boxes = detection_boxes / np.array([w, h, w, h])
        detection_boxes = detection_boxes * np.array([image_width, image_height, image_width, image_height])

        if _groundtruth_boxes is not None:
            gt_boxes = _groundtruth_boxes[i]
            gt_boxes = gt_boxes * np.array([image_height, image_width, image_height, image_width])
        else:
            gt_boxes = None

        if _detection_masks is not None:
            instance_masks = _detection_masks[i][0:num_detections]
            segmentations = coco_metric.generate_segmentation_from_masks(
                instance_masks,
                detection_boxes,
                image_height,
                image_width
            )
        else:
            segmentations = None

        # From [x, y, w, h] to [x1, y1, x2, y2] and
        # process_prediction_for_eval() set the box to be [x, y] format, need to
        # reverted them to [y, x] format.
        xmin, ymin, w, h = np.split(detection_boxes, 4, axis=-1)
        xmax = xmin + w
        ymax = ymin + h

        boxes_to_visualize = np.concatenate([ymin, xmin, ymax, xmax], axis=-1)

        image_preview = generate_image_preview(
            image,
            boxes=boxes_to_visualize,
            scores=detection_scores,
            classes=detection_classes.astype(np.int32),
            gt_boxes=gt_boxes,
            segmentations=segmentations
        )
        image_previews.append(image_preview)

    try:
        summaries = []

        for i, image_preview in enumerate(image_previews):
            image_buffer = generate_image_buffer(image_preview)
            image_summary = tf.compat.v1.Summary.Image(encoded_image_string=image_buffer)
            image_value = tf.compat.v1.Summary.Value(tag='%d_input' % i, image=image_summary)

            summaries.append(image_value)

    except AttributeError:
        image_previews = np.array(image_previews)
        summaries = tf.summary.image(
            name='image_summary',
            data=image_previews,
            step=current_step,
            max_outputs=max_images
        )

    return summaries

#@profile_dec 
def coco_box_eval(predictions, annotations_file, use_ext, use_dist_coco_eval):
    start = time.time()
    imgIds = []
    box_predictions = np.empty((len(predictions), 7))
    #print(predictions)
    for ii, prediction in enumerate(predictions):
      imgIds.append(prediction['image_id'])
      #box_predictions.append( [prediction['image_id']]+ list( map(lambda x: float(round(x, 2)), prediction['bbox'][:4])) + [float(prediction['score']), prediction['category_id']] )
      #box_predictions.append( [prediction['image_id']]+ np.array(prediction['bbox'][:4], dtype=np.float).round(2).tolist() + [float(prediction['score']), prediction['category_id']] )
      #box_predictions.append( [prediction['image_id']]+ prediction['bbox'][:4] + [float(prediction['score']), prediction['category_id']] )
      box_predictions[ii,0] = prediction['image_id']
      box_predictions[ii,1:5] = prediction['bbox'][:4] 
      box_predictions[ii, 5:]= [float(prediction['score']), prediction['category_id']]
    preproc_end = time.time()
    cocoGt = COCO(annotation_file=annotations_file, use_ext=use_ext)
    cocoDt = cocoGt.loadRes(box_predictions, use_ext=use_ext)
    #cocoDt = cocoGt.loadRes(predictions, use_ext=True)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox', use_ext=use_ext, num_threads=24)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate(dist=use_dist_coco_eval)
    cocoEval.summarize()
    print(f"Prepocessing box {preproc_end - start} coco c++ ext {time.time() - preproc_end}")

#@profile_dec 
def coco_mask_eval(predictions, annotations_file, use_ext, use_dist_coco_eval):
    start = time.time()
    imgIds = []
    for prediction in predictions:
      del prediction['bbox']
      imgIds.append(prediction['image_id'])
    preproc_end = time.time()
    cocoGt = COCO(annotation_file=annotations_file, use_ext=use_ext)
    cocoDt = cocoGt.loadRes(predictions, use_ext=use_ext)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='segm', use_ext=use_ext, num_threads=24)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate(dist=use_dist_coco_eval)
    cocoEval.summarize()
    print(f"Prepocessing mask {preproc_end - start} coco c++ ext {time.time() - preproc_end}")

def fast_eval(predictions, annotations_file, use_ext, use_dist_coco_eval):
    import pickle
    with open("/tmp/coco_eval_b12", 'wb') as fp:
      pickle.dump(predictions, fp)

    imgIds = []
    box_predictions = np.empty((len(predictions), 7))
    for ii, prediction in enumerate(predictions):
      imgIds.append(prediction['image_id'])
      box_predictions[ii,0] = prediction['image_id']
      box_predictions[ii,1:5] = prediction['bbox'][:4] 
      box_predictions[ii, 5:]= [float(prediction['score']), prediction['category_id']]
      del prediction['bbox']

    imgIds = list(set(imgIds))
    print(use_ext, use_dist_coco_eval, len(imgIds), len(predictions), flush=True)

    #BBox
    cocoGt = COCO(annotation_file=annotations_file, use_ext=use_ext)
    cocoDt = cocoGt.loadRes(box_predictions, use_ext=use_ext)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox', use_ext=use_ext, num_threads=24)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate(dist=use_dist_coco_eval)
    
    #Segm
    scocoDt = cocoGt.loadRes(predictions, use_ext=use_ext)
    scocoEval = COCOeval(cocoGt, cocoDt, iouType='segm', use_ext=use_ext, num_threads=24)
    scocoEval.params.imgIds = imgIds
    scocoEval.evaluate(dist=use_dist_coco_eval)
    if(MPI_rank() == 0):
      cocoEval.accumulate()
      scocoEval.accumulate()
      logging.info("Bbox Summary")
      cocoEval.summarize()
      logging.info("Segm Summary")
      scocoEval.summarize()
      
    
    return
  