#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="casia_vgg_$now"
REGION="us-central1"

BASE_DIR="gs://two-face-inpainting-mlengine/experiments"
EXPERIMENT_NAME="two_face_v6_1_facenet"
EXPERIMENT_DIR="$BASE_DIR/$EXPERIMENT_NAME"

# DATASET_PATH="gs://two-face-inpainting-mlengine/eval_data"
DATASET_PATH="gs://two-face-inpainting-mlengine/eval_data/val_data.zip"

FACENET_DIR="gs://two-face-inpainting-mlengine/facenet"

echo $EXPERIMENT_DIR

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $EXPERIMENT_DIR \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    --scale-tier BASIC \
    --runtime-version 1.12 \
    --python-version 3.5 \
    -- \
    --dataset_path $DATASET_PATH \
    --experiment_dir $EXPERIMENT_DIR \
    --facenet_dir $FACENET_DIR \
    --run_mode "EVAL" \
    --verbosity "INFO" \
    \
    --batch_normalization \
    --batch_size 64 \
    --lambda_rec 1.0 \
    --lambda_adv_local 0.01 \
    --lambda_adv_global 0.005 \
    --lambda_id 0.001 \
    --lambda_local_disc 0.1 \
    --lambda_global_disc 0.1
