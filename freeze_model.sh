#!/usr/bin/env bash

PIPELINE_CONFIG_PATH=$1
TRAINED_CKPT_PREFIX=$2
EXPORT_DIR=$3

ROOT_DIR=/localhome/rubenpurdy
cd $ROOT_DIR/tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf

INPUT_TYPE=image_tensor
python3 object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
