#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# run.sh — Orchestrates the full YOLOv10 → NCNN conversion pipeline
#
# Usage: ./run.sh [model_size] [rpi_user@rpi_host]
#   model_size     : yolov10n (default), yolov10s, yolov10m, etc.
#   rpi_user@host  : optional, for the deploy step (e.g. pi@raspberrypi.local)
###############################################################################

MODEL="${1:-yolov10n}"
RPI_TARGET="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  YOLOv10 → NCNN  (model=${MODEL})"
echo "============================================"

echo ""
echo "[lint] Running shellcheck…"
bash "${SCRIPT_DIR}/scripts/lint.sh"

echo ""
echo "[0/3] Setting up dependencies…"
bash "${SCRIPT_DIR}/scripts/0_setup.sh"

echo ""
echo "[1/3] Exporting ${MODEL} to ONNX…"
bash "${SCRIPT_DIR}/scripts/1_export_onnx.sh" "${MODEL}"

echo ""
echo "[2/3] Converting ONNX → NCNN via PNNX…"
bash "${SCRIPT_DIR}/scripts/2_pnnx_convert.sh" "${MODEL}"

echo ""
echo "[3/3] Deploying to Raspberry Pi 5…"
if [[ -n "${RPI_TARGET}" ]]; then
    bash "${SCRIPT_DIR}/scripts/3_deploy_rpi5.sh" "${MODEL}" "${RPI_TARGET}"
else
    echo "  ⏭  Skipped — no RPi target provided."
    echo "     Run manually: bash scripts/3_deploy_rpi5.sh ${MODEL} user@host"
fi

echo ""
echo "============================================"
echo "  ✅ Pipeline complete"
echo "  Artifacts: output/${MODEL}.*"
echo "============================================"
