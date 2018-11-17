#!/bin/bash

MODEL_DIR="checkpoints"
rm -rf $MODEL_DIR/*
DATASET_PATH="/home/gaston/workspace/datasets/CASIA-WebFace/CASIA-WebFace"
# DATASET_PATH="gs://two-face-inpainting-mlengine/data"

gcloud ml-engine local train \
    --job-dir $MODEL_DIR \
    --module-name trainer.task \
    --package-path trainer/ \
    -- \
    --dataset_path $DATASET_PATH \
    --checkpoints_dir $MODEL_DIR \
    --verbosity "INFO"