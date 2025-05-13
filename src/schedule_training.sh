#!/usr/bin/env bash
set -e  # Exit on any error

CONFIG_DIR="$1"

if [ -z "$CONFIG_DIR" ]; then
    echo "Usage: $0 <config_directory>"
    exit 1
fi

for config in "$CONFIG_DIR"/*.yaml; do
    echo "Running training with config: $config"
    python -u train.py --config "$config" --name test
done