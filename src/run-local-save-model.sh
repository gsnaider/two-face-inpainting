#!/bin/bash

#EXPERIMENT_DIR="/home/gaston/workspace/two-face/two-face-inpainting-experiments/local-runs/checkpoints"
#rm -rf $EXPERIMENT_DIR/*

#EXPERIMENT_DIR="/home/gaston/workspace/two-face/two-face-inpainting-experiments/trained-models/model_v6_1/two_face_v6_1_facenet"
EXPERIMENT_DIR="/home/gaston/workspace/two-face/two-face-inpainting-experiments/trained-models/model_v6_1_facenet_x10/two_face_v6_1_facenet_2"

#BASE_DIR="gs://two-face-inpainting-mlengine/experiments"
#EXPERIMENT_NAME="casia_vgg_rec_only_v4_5_2"
#EXPERIMENT_DIR="$BASE_DIR/$EXPERIMENT_NAME"

#EXPERIMENT_DIR="/home/gaston/workspace/two-face/two-face-inpainting-experiments/local-runs/save_model_test_3"
#EXPERIMENT_DIR="/home/gaston/workspace/two-face/two-face-inpainting-experiments/trained-models/casia_vgg_rec_local_adv_global_adv_facenet_v4_5"
#EXPERIMENT_DIR="/home/gaston/workspace/two-face/two-face-inpainting-experiments/trained-models/casia_vgg_rec_local_global_adv_8"

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
    --run_mode "SAVE_MODEL" \
    --model_number 1 \
    --verbosity "INFO" \
    \
    --batch_normalization