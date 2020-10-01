#!/bin/bash

mkdir -p ~/data/nv_coco
cd ~/data/nv_coco
aws s3 cp --recursive s3://jbsnyder-sagemaker/data/coco/nv_coco/ .
cd ..
mkdir annotations
cd annotations
aws s3 cp --recursive s3://jbsnyder-sagemaker/data/coco/annotations .

cd
git clone -b ablation https://github.com/johnbensnyder/DeepLearningExamples/
cd DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/
./download_and_process_pretrained_weights.sh
cd 

conda create --name mask_rcnn --clone tensorflow2_latest_p37
source activate mask_rcnn
pip install tensorflow_addons
pip install pybind11
git clone https://github.com/NVIDIA/cocoapi
cd cocoapi/PythonAPI
make install
conda install -y mpi4py
pip install git+https://github.com/NVIDIA/dllogger.git

