#!/bin/bash
docker run -t -d \
  -v `pwd`/../mimic_sum_data/:/mimic_sum_data/ \
  -v `pwd`/model_cfg:/home/experiment_cfg/ \
  -v hf-model-cache:/home/hf-model-cache/ \
  -v model-outputs:/home/model-outputs/ \
  -p  9000:9000 tsearle/mimic_summ:t5-small bash jupyter_start.sh
