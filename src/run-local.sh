#!/bin/bash

MODEL_DIR="/home/gaston/workspace/two-face/two-face-inpainting-experiments/local-runs/checkpoints"
rm -rf $MODEL_DIR/*

DATASET_PATH="/home/gaston/workspace/datasets/CASIA-WebFace/CASIA-WebFace/data"
# DATASET_PATH="/home/gaston/workspace/datasets/CASIA-WebFace/CASIA-WebFace/data/data.zip"
# DATASET_PATH="gs://two-face-inpainting-mlengine/sample-data"

FACENET_DIR="/home/gaston/workspace/two-face/facenet"

gcloud ml-engine local train \
    --job-dir $MODEL_DIR \
    --module-name trainer.task \
    --package-path trainer/ \
    -- \
    --dataset_path $DATASET_PATH \
    --experiment_dir $MODEL_DIR \
    --facenet_dir $FACENET_DIR \
    --batch_size 16 \
    --train \
    --verbosity "INFO"
