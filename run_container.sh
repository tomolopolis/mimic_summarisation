#!/bin/bash
docker run -t -d --name mimic_sum_jupyter -v "$(pwd)"/../mimic_sum_data/:/app -p 9000:9000 tsearle/mimic_sum:latest jupyter_start.sh
