#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG_PATH=${1:-training/configs/configs.yaml}
RUN_NAME="ray_run_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="training/logs/terminal"
mkdir -p "${LOG_DIR}"

echo "🚀 Starting Ray training with: ${CONFIG_PATH}"
python -m training.ray_main --config_path "${CONFIG_PATH}" | tee "${LOG_DIR}/${RUN_NAME}.log"
echo "✅ Training completed: ${RUN_NAME}"