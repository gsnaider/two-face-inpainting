#!/bin/bash

MODEL_DIR=/home/gaston/workspace/two-face/two-face-inpainting-experiments/trained-models/two_face_v6_rec_loss/saved_model

sudo docker run --runtime=nvidia -p 8501:8501   --mount type=bind,source=$MODEL_DIR,target=/models/two_face_model   -e MODEL_NAME=two_face_model -t tensorflow/serving:latest-gpu &
