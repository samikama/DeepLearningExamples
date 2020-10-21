#!/usr/bin/env bash
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


rm -rf results_tf2_1x_novo_giou_cosine_carl
mkdir -p results_tf2_1x_novo_giou_cosine_carl
mpirun --allow-run-as-root --tag-output --mca plm_rsh_no_tree_spawn 1 \
    --mca btl_tcp_if_exclude lo,docker0 \
    -np 8 -H localhost:8 \
    -x NCCL_DEBUG=VERSION \
    -x LD_LIBRARY_PATH \
    -x PATH \
    --oversubscribe \
    python ../mask_rcnn_main.py \
        --mode="train_and_eval" \
	--loop_mode="tape" \
        --checkpoint="/MaskRCNN.giou/weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" \
        --eval_samples=5000 \
        --init_learning_rate=0.008 \
        --optimizer_type="Novograd" \
	--box_loss_type="giou" \
        --lr_schedule="cosine" \
	--use_carl_loss \
        --model_dir="results_tf2_1x_novo_giou_cosine_carl" \
        --num_steps_per_eval=3696 \
        --warmup_learning_rate=0.000133 \
	--beta1=0.9 \
	--beta2=0.4 \
	--warmup_steps=100 \
        --total_steps=45000 \
	--learning_rate_steps=30000,40000 \
        --l2_weight_decay=5e-4 \
        --train_batch_size=4 \
        --eval_batch_size=1 \
        --dist_eval \
        --training_file_pattern="/nv_tfrecords/train*.tfrecord" \
        --validation_file_pattern="/nv_tfrecords/val*.tfrecord" \
        --val_json_file="/nv_tfrecords/annotations/instances_val2017.json" \
        --amp \
        --use_batched_nms \
        --xla \
        --tf2 \
        --use_custom_box_proposals_op | tee results_tf2_1x_novo_giou_cosine_carl/results_tf2_1x_novo.log

