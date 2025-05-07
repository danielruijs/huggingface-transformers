#!/usr/bin/env bash

MODEL_DIR="~/huggingface-transformers/rtdetrv2"
LOG_FILE="train.log"

cd $MODEL_DIR
rm -f "${LOG_FILE}" 2>/dev/null || true
echo "Starting Training. Logging output to: ${LOG_FILE}"

# Start training with nohup
nohup python train.py > "${LOG_FILE}" 2>&1 &
tail -f "${LOG_FILE}"