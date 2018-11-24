#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="casia_vgg_rec_only_$now"
BASE_DIR="gs://two-face-inpainting-mlengine/experiments"
REGION="us-central1"
EXPERIMENT_DIR="$BASE_DIR/$JOB_NAME"

# DATASET_PATH="gs://two-face-inpainting-mlengine/data"
DATASET_PATH="gs://two-face-inpainting-mlengine/data/data.zip"

FACENET_DIR="gs://two-face-inpainting-mlengine/facenet"

echo $EXPERIMENT_DIR

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $EXPERIMENT_DIR \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    --scale-tier BASIC_GPU \
    --runtime-version 1.10 \
    --python-version 3.5 \
    -- \
    --dataset_path $DATASET_PATH \
    --experiment_dir $EXPERIMENT_DIR \
    --facenet_dir $FACENET_DIR \
    --batch_size 32 \
    --verbosity "INFO"
