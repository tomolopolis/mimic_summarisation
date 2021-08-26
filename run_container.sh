#!/bin/bash
# runs the container in offline mode - remove --env-file otherwise
docker run -t -d --env-file offline-env.env --gpus all \
  -v `pwd`/../mimic_summ_data/:/mimic_summ_data/ \
  -v experiment-cfg:/home/experiment_cfg/ \
  -v model-outputs:/home/model-outputs/ -p 9005:9005 \
  tsearle/mimic_summ:latest jupyter lab --port=9005 --no-browser --allow-root --ip=0.0.0.0
