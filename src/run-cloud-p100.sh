#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="casia_vgg_$now"
REGION="us-central1"

BASE_DIR="gs://two-face-inpainting-mlengine/experiments"

EXPERIMENT_NAME="two_face_v6_1_facenet_2"

EXPERIMENT_DIR="$BASE_DIR/$EXPERIMENT_NAME"

# DATASET_PATH="gs://two-face-inpainting-mlengine/data"
DATASET_PATH="gs://two-face-inpainting-mlengine/data/data.zip"

FACENET_DIR="gs://two-face-inpainting-mlengine/facenet"

echo $EXPERIMENT_DIR

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $EXPERIMENT_DIR \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    --config config-p100.yaml \
    --runtime-version 1.12 \
    --python-version 3.5 \
    -- \
    --dataset_path $DATASET_PATH \
    --experiment_dir $EXPERIMENT_DIR \
    --facenet_dir $FACENET_DIR \
    --run_mode "TRAIN" \
    --verbosity "INFO" \
    \
    --batch_normalization \
    --batch_size 64 \
    --max_steps 40e3 \
    --gen_learning_rate 1e-6 \
    --disc_learning_rate 1e-6 \
    --lambda_rec 1.0 \
    --lambda_adv_local 0.01 \
    --lambda_adv_global 0.005 \
    --lambda_id 0.01 \
    --lambda_local_disc 0.1 \
    --lambda_global_disc 0.1
