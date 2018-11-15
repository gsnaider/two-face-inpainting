#!/bin/bash

MODEL_DIR="checkpoints"
rm -rf $MODEL_DIR/*
echo $MODEL_DIR

gcloud ml-engine local train \
    --job-dir $MODEL_DIR \
    --module-name trainer.task \
    --package-path trainer/ \
