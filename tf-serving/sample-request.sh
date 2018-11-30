#!/bin/bash

OUTPUT_FILE=patched_img

curl -d "@sample_params" -X POST http://localhost:8501/v1/models/two_face_model:predict > $OUTPUT_FILE