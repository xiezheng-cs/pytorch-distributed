#!/usr/bin/env bash
nvidia-smi -i 0,1,2,3 --format=csv,noheader,nounits --query-gpu=timestamp,index,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory -lms 500 -f distributed_log.csv
nvidia-smi -i 0,1,2,3 --format=csv,noheader,nounits --query-gpu=timestamp,index,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory -lms 500 -f apex_distributed_log.csv
