#!/bin/bash

MODEL_DIR=/home/gaston/workspace/two-face/two-face-inpainting-experiments/trained-models/model_v6_1/two_face_v6_1_facenet/saved_model

echo $MODEL_DIR
sudo docker run --runtime=nvidia -p 8501:8501   --mount type=bind,source=$MODEL_DIR,target=/models/two_face_model   -e MODEL_NAME=two_face_model -t tensorflow/serving:latest-gpu &
