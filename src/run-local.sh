#!/bin/bash

MODEL_DIR="checkpoints"
rm -rf $MODEL_DIR/*
echo $MODEL_DIR

gcloud ml-engine local train \
    --job-dir $MODEL_DIR \
    --module-name trainer.task \
    --package-path trainer/ \
    -- \
    --dataset_path "/home/gaston/workspace/datasets/CASIA-WebFace/CASIA-WebFace" \
    --checkpoints_dir "/home/gaston/workspace/two-face-inpainting/src/checkpoints"
