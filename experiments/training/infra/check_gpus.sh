#!/bin/bash
# Projeto IANA — Status resumido das GPUs
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
