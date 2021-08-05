#!/bin/bash
docker run -t -d \
  --env-file offline-env.env \
  -v "$(pwd)"/../mimic_sum_data/:/mimic_sum_data/ \
  -v hf-model-cache:/home/hf-model-cache/ \
  -v model-outputs:/home/model-outputs/ \
  -p  9000:9000 tsearle/mimic_sum:latest jupyter_start.sh
