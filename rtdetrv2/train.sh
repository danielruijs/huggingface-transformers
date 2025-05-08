#!/usr/bin/env bash

LOG_FILE="train.log"

rm -f "${LOG_FILE}" 2>/dev/null || true
echo "Starting Training. Logging output to: ${LOG_FILE}"

# Start training with nohup
nohup python train.py > "${LOG_FILE}" 2>&1 &
tail -f "${LOG_FILE}"