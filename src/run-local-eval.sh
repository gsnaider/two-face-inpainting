#!/bin/bash

EXPERIMENT_DIR="/home/gaston/workspace/two-face/two-face-inpainting-experiments/local-runs/checkpoints"
#rm -rf $EXPERIMENT_DIR/*

#EXPERIMENT_DIR="/home/gaston/workspace/two-face/two-face-inpainting-experiments/trained-models/full_trained"

#BASE_DIR="gs://two-face-inpainting-mlengine/experiments"
#EXPERIMENT_NAME="casia_vgg_rec_only_v4_5_2"
#EXPERIMENT_DIR="$BASE_DIR/$EXPERIMENT_NAME"

#EXPERIMENT_DIR="/home/gaston/workspace/two-face/two-face-inpainting-experiments/trained-models/casia_vgg_rec_local_adv_global_adv_facenet_v4_5"

#EXPERIMENT_DIR="/home/gaston/workspace/two-face/two-face-inpainting-experiments/trained-models/full_trained"


DATASET_PATH="/home/gaston/workspace/datasets/CASIA-WebFace/CASIA-WebFace/data/val_data"
# DATASET_PATH="/home/gaston/workspace/datasets/CASIA-WebFace/CASIA-WebFace/data/val_data/val_data.zip"
# DATASET_PATH="gs://two-face-inpa  inting-mlengine/sample-data"

FACENET_DIR="/home/gaston/workspace/two-face/facenet"

gcloud ml-engine local train \
    --job-dir $EXPERIMENT_DIR \
    --module-name trainer.task \
    --package-path trainer/ \
    -- \
    --dataset_path $DATASET_PATH \
    --experiment_dir $EXPERIMENT_DIR \
    --facenet_dir $FACENET_DIR \
    --run_mode "EVAL" \
    --verbosity "INFO" \
    \
    --batch_normalization \
    --batch_size 8 \
    --lambda_rec 1.0 \
    --lambda_adv_local 0.0 \
    --lambda_adv_global 0.0 \
    --lambda_id 0.0 \
    --lambda_local_disc 0.0 \
    --lambda_global_disc 0.0
