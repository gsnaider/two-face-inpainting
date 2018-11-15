#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="casia_vgg_$now"
JOB_DIR="gs://first-ml-project-222122-mlengine"
REGION="us-east1"
OUTPUT_PATH="$JOB_DIR/$JOB_NAME"

echo $OUTPUT_PATH

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    --scale-tier BASIC_GPU \
    --runtime-version 1.10 \
    --python-version 3.5