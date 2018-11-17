#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="casia_vgg_$now"
JOB_DIR="gs://two-face-inpainting-mlengine/experiments"
REGION="us-central1"
OUTPUT_PATH="$JOB_DIR/$JOB_NAME"
DATASET_PATH="gs://two-face-inpainting-mlengine/data"

echo $OUTPUT_PATH

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    --scale-tier BASIC_GPU \
    --runtime-version 1.10 \
    --python-version 3.5 \
    -- \
    --dataset_path $DATASET_PATH \
    --checkpoints_dir $OUTPUT_PATH \
    --verbosity "INFO"
