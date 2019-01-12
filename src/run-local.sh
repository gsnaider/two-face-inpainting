#!/bin/bash

EXPERIMENT_DIR="/home/gaston/workspace/two-face/two-face-inpainting-experiments/local-runs/checkpoints"
rm -rf $EXPERIMENT_DIR/*

#EXPERIMENT_DIR="/home/gaston/workspace/two-face/two-face-inpainting-experiments/trained-models/full_trained"

DATASET_PATH="/home/gaston/workspace/datasets/CASIA-WebFace/CASIA-WebFace/data/train_data"
# DATASET_PATH="/home/gaston/workspace/datasets/CASIA-WebFace/CASIA-WebFace/data/train_data/data.zip"
# DATASET_PATH="gs://two-face-inpainting-mlengine/sample-data"

FACENET_DIR="/home/gaston/workspace/two-face/facenet"

gcloud ml-engine local train \
    --job-dir $EXPERIMENT_DIR \
    --module-name trainer.task \
    --package-path trainer/ \
    -- \
    --dataset_path $DATASET_PATH \
    --experiment_dir $EXPERIMENT_DIR \
    --facenet_dir $FACENET_DIR \
    --batch_size 8 \
    --run_mode "TRAIN" \
    --verbosity "INFO"
