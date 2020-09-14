import tensorflow.compat.v1 as tf
import tensorflow as tf2
import json
import logging
import sys
import os
sys.path.append('..')
from mask_rcnn.anchors import Anchors, AnchorLabeler
from mask_rcnn.ops import preprocess_ops
from pycocotools import mask
import numpy as np

import horovod.tensorflow as hvd

from mask_rcnn.utils.logging_formatter import logging

from mask_rcnn.utils.distributed_utils import MPI_is_distributed
from mask_rcnn.utils.distributed_utils import MPI_rank_and_size
from mask_rcnn.utils.distributed_utils import MPI_rank
from mask_rcnn.utils.distributed_utils import MPI_size
from distutils.version import LooseVersion

def create_category_index(categories):
    """Creates dictionary of COCO compatible categories keyed by category id.
    Args:
        categories: a list of dicts, each of which has the following keys:
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.
    Returns:
        category_index: a dict containing the same entries as categories, but keyed
        by the 'id' field of each category.
    """
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index

def parse_annotations(annotations_file):
    with tf.io.gfile.GFile(annotations_file, 'r') as fid:
        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']
        category_index = create_category_index(
            groundtruth_data['categories'])
    annotations_index = {}
    if 'annotations' in groundtruth_data:
        logging.info('Found groundtruth annotations. Building annotations index.')
        for annotation in groundtruth_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_index:
                annotations_index[image_id] = []
            annotations_index[image_id].append(annotation)
    missing_annotation_count = 0
    for image in images:
        image_id = image['id']
        if image_id not in annotations_index:
            missing_annotation_count += 1
            annotations_index[image_id] = []
    logging.info('%d images are missing annotations.', missing_annotation_count)
    return images, category_index, annotations_index

class FastDataLoader(object):
    
    def __init__(self, file_pattern, params):
        self._file_pattern = file_pattern
        self.image_preprocess = PreprocessImage(params)
        self.params = params
    
    def __call__(self, params, input_context=None):
        batch_size = params['batch_size'] if 'batch_size' in params else 1
        try:
            seed = params['seed'] * hvd.rank()
        except (KeyError, TypeError):
            seed = None
        n_gpus = hvd.size()
            
        dataset = tf.data.Dataset.list_files(
            self._file_pattern,
            shuffle=False
        )
        if input_context is not None:
            logging.info("Using Dataset Sharding with TF Distributed")
            _num_shards = input_context.num_input_pipelines
            _shard_idx = input_context.input_pipeline_id
        
        logging.info("Using Dataset Sharding with Horovod")
        _shard_idx = hvd.rank() 
        _num_shards = hvd.size()
        
        try:
            dataset = dataset.shard(
                num_shards=_num_shards,
                index=_shard_idx
            )
            dataset = dataset.shuffle(math.ceil(256 / _num_shards))
        
        except NameError:  # Not a distributed training setup
            pass
        
        def _prefetch_dataset(filename):
            return tf.data.TFRecordDataset(filename).prefetch(1)

        dataset = dataset.interleave(
            map_func=_prefetch_dataset,
            cycle_length=32,
            block_length=64,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        
        dataset = dataset.cache()
        
        dataset = dataset.shuffle(
            buffer_size=4096,
            reshuffle_each_iteration=True,
            seed=seed
        )
        
        dataset = dataset.repeat()
        
        dataset = dataset.map(
            map_func=self.image_preprocess,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        
        dataset = dataset.batch(
            batch_size=batch_size,
            drop_remainder=True
        )
        
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE,
        )

        '''dataset = dataset.apply(
            tf.data.experimental.prefetch_to_device(
                '/gpu:{}'.format(hvd.rank()),  # With Horovod the local GPU is always 0
                buffer_size=1,
            )
        )'''

        data_options = tf.data.Options()
        
        data_options.experimental_deterministic = seed is not None
        if LooseVersion(tf.__version__) <= LooseVersion("2.0.0"):
            data_options.experimental_distribute.auto_shard = False
        else:
            data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        # data_options.experimental_distribute.auto_shard = False
        data_options.experimental_slack = True

        data_options.experimental_threading.max_intra_op_parallelism = 1
        # data_options.experimental_threading.private_threadpool_size = int(multiprocessing.cpu_count() / n_gpus) * 2

        # ================= experimental_optimization ================= #

        data_options.experimental_optimization.apply_default_optimizations = False

        # data_options.experimental_optimization.autotune = True
        data_options.experimental_optimization.filter_fusion = True
        data_options.experimental_optimization.map_and_batch_fusion = True
        data_options.experimental_optimization.map_and_filter_fusion = True
        data_options.experimental_optimization.map_fusion = True
        data_options.experimental_optimization.map_parallelization = True

        map_vectorization_options = tf.data.experimental.MapVectorizationOptions()
        map_vectorization_options.enabled = True
        map_vectorization_options.use_choose_fastest = True

        data_options.experimental_optimization.map_vectorization = map_vectorization_options

        data_options.experimental_optimization.noop_elimination = True
        data_options.experimental_optimization.parallel_batch = True
        data_options.experimental_optimization.shuffle_and_repeat_fusion = True

        # ========== Stats on TF Data =============
        # aggregator = tf.data.experimental.StatsAggregator()
        # data_options.experimental_stats.aggregator = aggregator
        # data_options.experimental_stats.latency_all_edges = True

        dataset = dataset.with_options(data_options)

        return dataset
        

class PreprocessImage(object):
    def __init__(self, params, MAX_NUM_INSTANCES=100):
        self.size = params['image_size']
        self.params = params
        self.MAX_NUM_INSTANCES = MAX_NUM_INSTANCES
    
    def __call__(self, example):
        features, labels = self.parse_example(example)
        image = tf.cast(features['images'], tf.float32)
        image = preprocess_ops.normalize_image(image/255.)
        image, image_size, padding = self.resize(image)
        # trying fp16
        #image = tf.cast(image, tf.float16)
        image.set_shape((832, 1344, 3))
        features['images'] = image
        labels['cropped_gt_masks'] = self.pad_masks(labels['cropped_gt_masks'])
        labels['gt_boxes'] = self.pad_to_fixed_size(labels['gt_boxes'], -1, 
                                                    [self.MAX_NUM_INSTANCES, 4])
        labels['gt_classes'] = self.pad_to_fixed_size(labels['gt_classes'], -1,
                                                      [self.MAX_NUM_INSTANCES, 1])
        #labels['cropped_gt_masks'] = tf.cast(labels['cropped_gt_masks'], tf.float16)
        #labels['gt_boxes'] = tf.cast(labels['gt_boxes'], tf.float16)
        #labels['gt_classes'] = tf.cast(labels['gt_classes'], tf.float16)
        anchors = self.deserialize_anchors(labels)
        labels.update(anchors)
        return features, labels
    
    def deserialize_anchors(self, labels):
        anchors = {}
        for lvl in range(self.params['min_level'], self.params['max_level']+1):
            anchors['score_targets_{}'.format(lvl)] = \
                self.deserialized_sparse(labels['score_targets_{}'.format(lvl)], 
                                         tf.int32, default=-1)
            anchors['box_targets_{}'.format(lvl)] = \
                self.deserialized_sparse(labels['box_targets_{}'.format(lvl)], 
                                         tf.float32, default=0.)
        return anchors
    
    def deserialized_sparse(self, tensor, dtype, default=None):
        expanded = tf.expand_dims(tensor, axis=0)
        sparse = tf.io.deserialize_many_sparse(expanded, dtype)
        dense = tf.squeeze(tf.sparse.to_dense(sparse, default_value=default))
        return dense
        
    def resize(self, image):
        resized_image = tf.image.resize(image, self.size, preserve_aspect_ratio=True)
        image_size = tf.shape(resized_image)[:2]
        padding = [[0, self.size[0]-tf.shape(resized_image)[0]], 
                   [0, self.size[1]-tf.shape(resized_image)[1]], [0,0]]
        resized_image = tf.pad(resized_image, padding)
        return resized_image, image_size, padding 
    
    def pad_masks(self, gt_masks):
        gt_masks = self.pad_to_fixed_size(gt_masks, -1, [self.MAX_NUM_INSTANCES, 
                                                         (self.params['gt_mask_size'] + 4) ** 2])
        gt_masks = tf.reshape(gt_masks, 
                              [self.MAX_NUM_INSTANCES, 
                               self.params['gt_mask_size'] + 4, 
                               self.params['gt_mask_size'] + 4])
        return gt_masks
        
    
    def pad_to_fixed_size(self, data, pad_value, output_shape):
        max_num_instances = output_shape[0]
        dimension = output_shape[1]
        data = tf.reshape(data, [-1, dimension])
        num_instances = tf.shape(input=data)[0]
        pad_length = max_num_instances - num_instances
        paddings = pad_value * tf.ones([pad_length, dimension])
        padded_data = tf.reshape(tf.concat([data, paddings], axis=0), output_shape)
        return padded_data
    
    def parse_example(self, example):
        parsed = tf.io.parse_single_example(example, features_decoder)
        features = {}
        labels = {}
        #features['images'] = tf.io.parse_tensor(parsed['image/image'], tf.float32)
        features['images'] = tf.image.decode_jpeg(parsed['image/image_encoded'])
        if tf.shape(features['images'])[-1]==1:
            features['images'] = tf.image.grayscale_to_rgb(features['images'])
        features['source_ids'] = parsed['image/source_id']
        features['image_info'] = tf.io.parse_tensor(parsed['image/image_info'], tf.float32)
        features['image_info'].set_shape(5)
        labels['cropped_gt_masks'] = tf.io.parse_tensor(parsed['label/cropped_gt_masks'], tf.float32)
        labels['gt_boxes'] = tf.io.parse_tensor(parsed['label/gt_boxes'], tf.float32)
        labels['gt_classes'] = tf.io.parse_tensor(parsed['label/gt_classes'], tf.float32)
        labels['score_targets_2'] = parsed['label/score_targets_2']
        labels['score_targets_3'] = parsed['label/score_targets_3']
        labels['score_targets_4'] = parsed['label/score_targets_4']
        labels['score_targets_5'] = parsed['label/score_targets_5']
        labels['score_targets_6'] = parsed['label/score_targets_6']
        labels['box_targets_2'] = parsed['label/box_targets_2']
        labels['box_targets_3'] = parsed['label/box_targets_3']
        labels['box_targets_4'] = parsed['label/box_targets_4']
        labels['box_targets_5'] = parsed['label/box_targets_5']
        labels['box_targets_6'] = parsed['label/box_targets_6']
        return features, labels

    
    
class PreprocessDataset(object):
    def __init__(self, params, image_dir, MAX_NUM_INSTANCES=100, include_crowd=False):
        self.size = params['image_size']
        self.image_dir = image_dir
        self.params = params
        self.include_crowd = include_crowd
        self.MAX_NUM_INSTANCES = MAX_NUM_INSTANCES
        self.anchors = Anchors(params['min_level'], 
                                         params['max_level'], 
                                         params['num_scales'], 
                                         params['aspect_ratios'], 
                                         params['anchor_scale'], 
                                         self.size)
        self.anchor_labeler = AnchorLabeler(self.anchors,
                                                      params['num_classes'])
        
    def __call__(self, image_data, annotations):
        image, image_encoded = self.read_image(image_data['file_name'])
        #aspect_ratio = image.shape[0]/image.shape[1]
        #size = self.landscape_size if aspect_ratio<1 else self.portrait_size
        original_size = image.shape[:2]
        image, image_size, padding = self.resize(image, self.size)
        image_info = self.get_image_info(original_size, image_size)
        gt_masks = self.get_masks(annotations, original_size, self.size)
        gt_boxes = self.get_boxes(annotations, image_info)
        gt_masks = self.crop_gt_masks(gt_masks, gt_boxes, self.size)
        gt_classes = self.get_classes(annotations)
        gt_classes = tf.cast(gt_classes, tf.float32)
        process_data = {'image': image,
                        'image_encoded': tf.convert_to_tensor(image_encoded),
                        'original_size': tf.convert_to_tensor(original_size), 
                        'image_info': image_info,
                        'gt_masks': gt_masks,
                        'image_id': tf.convert_to_tensor(image_data['id']),
                        'file_name': tf.convert_to_tensor(image_data['file_name'])}
        process_data.update(self.process_labels_for_training(image_info, 
                                                             gt_boxes, 
                                                             gt_classes))
        return process_data
    
    def process_labels_for_training(self, image_info, gt_boxes, gt_classes):
        labels = {}
        score_targets, box_targets = self.anchor_labeler.label_anchors(gt_boxes, gt_classes)
        for level in range(self.params['min_level'], self.params['max_level'] + 1):
            labels['score_targets_%d' % level] = score_targets[level]
            labels['box_targets_%d' % level] = box_targets[level]
        labels['gt_boxes'] = gt_boxes
        labels['gt_classes'] = gt_classes
        return labels
        
    def pad_to_fixed_size(self, data, pad_value, output_shape):
        max_num_instances = output_shape[0]
        dimension = output_shape[1]
        data = tf.reshape(data, [-1, dimension])
        num_instances = tf.shape(input=data)[0]
        pad_length = max_num_instances - num_instances
        paddings = pad_value * tf.ones([pad_length, dimension])
        padded_data = tf.reshape(tf.concat([data, paddings], axis=0), output_shape)
        return padded_data
    
    def get_image_info(self, original_size, image_size):
        scale_factor = image_size[0]/original_size[0]
        image_info = tf.stack([
            tf.cast(scale_factor, dtype=tf.float32),
            tf.cast(scale_factor, dtype=tf.float32),
            tf.cast(1.0/scale_factor, dtype=tf.float32),
            tf.cast(original_size[0], dtype=tf.float32),
            tf.cast(original_size[1], dtype=tf.float32)])
        return image_info
    
    def read_image(self, image_file):
        image_encoded = tf.io.gfile.GFile(os.path.join(self.image_dir, image_file),
                                          mode='rb').read()
        image = tf.image.decode_jpeg(image_encoded)
        return image, image_encoded
    
    def resize(self, image, size):
        resized_image = tf.image.resize(image, size, preserve_aspect_ratio=True)
        image_size = resized_image.shape[:2]
        padding = [[0, size[0]-resized_image.shape[0]], [0, size[1]-resized_image.shape[1]], [0,0]]
        resized_image = tf.pad(resized_image, padding)
        return resized_image, image_size, padding
    
    def get_classes(self, annotations):
        gt_classes = []
        for annot in annotations: 
            if self.include_crowd:
                gt_classes.append(annot['category_id'])
            elif not annot['iscrowd']:
                gt_classes.append(annot['category_id'])
        gt_classes = tf.expand_dims(tf.convert_to_tensor(gt_classes), axis=1)
        return gt_classes
    
    def get_masks(self, annotations, original_size, size):
        def get_single_mask(polygon, image_size, size):
            run_len_encoding = mask.frPyObjects(polygon, original_size[0], original_size[1])
            binary_mask = mask.decode(run_len_encoding)
            # if crowd, expand dim
            if binary_mask.ndim==2:
                binary_mask = np.expand_dims(binary_mask, -1)
            # take max across last axis to deal with multiple ploygons
            binary_mask = np.expand_dims(np.max(binary_mask, axis=-1), axis=-1)
            binary_mask, _, _ = self.resize(binary_mask, size)
            binary_mask = tf.convert_to_tensor(binary_mask, tf.float32)
            return binary_mask
        masks = []
        for annot in annotations:
            if self.include_crowd:
                masks.append(get_single_mask(annot['segmentation'], original_size, size))
            elif not annot['iscrowd']:
                masks.append(get_single_mask(annot['segmentation'], original_size, size))
        masks = tf.expand_dims(tf.transpose(tf.concat(masks, axis=2), [2,0,1]), axis=-1)
        return masks
    
    def crop_gt_masks(self, instance_masks, boxes, image_size):
        """Crops the ground truth binary masks and resize to fixed-size masks."""
        num_masks = tf.shape(input=instance_masks)[0]

        scale_sizes = tf.convert_to_tensor(value=[image_size[0], image_size[1]] * 2, dtype=tf.float32)

        boxes = boxes / scale_sizes

        cropped_gt_masks = tf.image.crop_and_resize(
            image=instance_masks,
            boxes=boxes,
            box_indices=tf.range(num_masks, dtype=tf.int32),
            crop_size=[self.params['gt_mask_size'], self.params['gt_mask_size']],
            method='bilinear')[:, :, :, 0]

        cropped_gt_masks = tf.pad(
            tensor=cropped_gt_masks,
            paddings=tf.constant([[0, 0], [2, 2], [2, 2]]),
            mode='CONSTANT',
            constant_values=0.
        )

        return cropped_gt_masks
    
    def get_boxes(self, annotations, image_info):
        gt_boxes = []
        for annot in annotations:
            if self.include_crowd:
                gt_boxes.append(tf.cast(tf.convert_to_tensor(annot['bbox']), tf.float32) * image_info[0])
            elif not annot['iscrowd']:
                gt_boxes.append(tf.cast(tf.convert_to_tensor(annot['bbox']), tf.float32) * image_info[0])
        gt_boxes = tf.stack(gt_boxes)
        gt_boxes = tf.transpose(tf.stack([gt_boxes[:,1],
                                           gt_boxes[:,0],
                                           gt_boxes[:,1] + gt_boxes[:,3],
                                           gt_boxes[:,0] + gt_boxes[:,2]]))
        return gt_boxes

def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.
    Args:
    exit_stack: A context2.ExitStack used to automatically closed the TFRecords
      opened in this function.
    base_path: The base path for all shards
    num_shards: The number of shards
    Returns:
    The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
      '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
      for idx in range(num_shards)
    ]

    tfrecords = [
      exit_stack.enter_context(tf.io.TFRecordWriter(file_name))
      for file_name in tf_record_output_filenames
    ]

    return tfrecords

def serialize_tensor(tensor):
    serialized = tf.io.serialize_tensor(tensor).numpy()
    serialized = tf.train.BytesList(value=[serialized])
    return tf.train.Feature(bytes_list=serialized)

def serialize_sparse(tensor, default_value):
    indices = tf.where(tensor!=default_value)
    values = tf.gather_nd(tensor, indices)
    sparse_tensor = tf.SparseTensor(indices, values, dense_shape=tensor.shape)
    serialized = tf.io.serialize_sparse(sparse_tensor).numpy()
    serialized = tf.train.BytesList(value=serialized)
    return tf.train.Feature(bytes_list=serialized)

def create_record(image_num, images, annotations_index, image_preprocess):
    annotations = annotations_index[images[image_num]['id']]
    if not annotations:
        return None, None
    result = image_preprocess(images[image_num], annotations)
    shape = result['image'].shape[:2]
    feature_dict = {
        'image/image_encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[result['image_encoded'].numpy()])),
        'image/source_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[result['image_id'].numpy()])),
        'image/image_info': serialize_tensor(result['image_info']),
        'label/cropped_gt_masks': serialize_tensor(result['gt_masks']),
        'label/gt_boxes': serialize_tensor(result['gt_boxes']),
        'label/gt_classes': serialize_tensor(result['gt_classes']),
        'label/score_targets_2': serialize_sparse(result['score_targets_2'], -1),
        'label/score_targets_3': serialize_sparse(result['score_targets_3'], -1),
        'label/score_targets_4': serialize_sparse(result['score_targets_4'], -1),
        'label/score_targets_5': serialize_sparse(result['score_targets_5'], -1),
        'label/score_targets_6': serialize_sparse(result['score_targets_6'], -1),
        'label/box_targets_2': serialize_sparse(result['box_targets_2'], 0),
        'label/box_targets_3': serialize_sparse(result['box_targets_3'], 0),
        'label/box_targets_4': serialize_sparse(result['box_targets_4'], 0),
        'label/box_targets_5': serialize_sparse(result['box_targets_5'], 0),
        'label/box_targets_6': serialize_sparse(result['box_targets_6'], 0),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, shape

features_decoder = {
    'image/image_encoded' : tf.io.FixedLenFeature((), tf.string),
    'image/source_id': tf.io.FixedLenFeature((), tf.int64),
    'image/image_info': tf.io.FixedLenFeature((), tf.string),
    'label/cropped_gt_masks': tf.io.FixedLenFeature((), tf.string),
    'label/gt_boxes': tf.io.FixedLenFeature((), tf.string),
    'label/gt_classes': tf.io.FixedLenFeature((), tf.string),
    'label/score_targets_2': tf.io.FixedLenFeature([3], tf.string),
    'label/score_targets_3': tf.io.FixedLenFeature([3], tf.string),
    'label/score_targets_4': tf.io.FixedLenFeature([3], tf.string),
    'label/score_targets_5': tf.io.FixedLenFeature([3], tf.string),
    'label/score_targets_6': tf.io.FixedLenFeature([3], tf.string),
    'label/box_targets_2': tf.io.FixedLenFeature([3], tf.string),
    'label/box_targets_3': tf.io.FixedLenFeature([3], tf.string),
    'label/box_targets_4': tf.io.FixedLenFeature([3], tf.string),
    'label/box_targets_5': tf.io.FixedLenFeature([3], tf.string),
    'label/box_targets_6': tf.io.FixedLenFeature([3], tf.string),
}

