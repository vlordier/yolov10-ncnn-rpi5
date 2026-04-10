#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 2_pnnx_convert.sh — Convert YOLOv10 PyTorch → TorchScript → NCNN via PNNX
#
# PNNX works best from TorchScript (not ONNX). This script exports directly
# from the Ultralytics .pt weights to TorchScript, then runs PNNX.
#
# Usage: bash scripts/2_pnnx_convert.sh [model_size]
###############################################################################

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output"
NCNN_DIR="${REPO_ROOT}/ncnn"
MODEL="${1:-yolov10n}"

PNNX_TOOL="${NCNN_DIR}/tools/pnnx/build/src/pnnx"

# Fallback paths
if [[ ! -x "${PNNX_TOOL}" ]]; then
    PNNX_TOOL="${NCNN_DIR}/tools/pnnx/build/pnnx"
fi
if [[ ! -x "${PNNX_TOOL}" ]]; then
    PNNX_TOOL="${NCNN_DIR}/tools/pnnx/build/src/pnnx.exe"
fi
TS_FILE="${OUTPUT_DIR}/${MODEL}.torchscript.pt"

# ── Validate PNNX ────────────────────────────────────────────────────────────
if [[ ! -x "${PNNX_TOOL}" ]]; then
    echo "ERROR: PNNX tool not found or not executable: ${PNNX_TOOL}"
    echo "       Run 0_setup.sh first."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

# ── Step 1: PyTorch → TorchScript ────────────────────────────────────────────
if [[ ! -f "${TS_FILE}" ]]; then
    echo "→ Exporting ${MODEL} to TorchScript…"
    # Run export from repo root so ultralytics puts output in expected location
    cd "${REPO_ROOT}"
    python3 -c "
import torch
from ultralytics import YOLO

model = YOLO('${MODEL}.pt')
model.export(format='torchscript', imgsz=640, simplify=True)
print('  ✓ TorchScript export complete')
"
    # Find the exported file and move to output dir
    cd "${REPO_ROOT}"
    for p in runs/detect/train*/weights/best.torchscript runs/detect/train*/weights/*.torchscript; do
        if [[ -f "$p" ]]; then
            mv "$p" "${TS_FILE}"
            echo "  Moved ${p} → ${TS_FILE}"
            break
        fi
    done
    # Also check for direct *.torchscript output
    for p in "${MODEL}"*.torchscript; do
        if [[ -f "$p" ]] && [[ "$p" != "${TS_FILE}" ]]; then
            mv "$p" "${TS_FILE}"
            echo "  Moved ${p} → ${TS_FILE}"
            break
        fi
    done
    cd "${OUTPUT_DIR}"
else
    echo "  ✓ TorchScript already exists: ${TS_FILE}"
fi

if [[ ! -f "${TS_FILE}" ]]; then
    echo "  ERROR: TorchScript export failed."
    exit 1
fi

# ── Step 2: TorchScript → NCNN via PNNX ──────────────────────────────────────
echo ""
echo "→ Running PNNX: TorchScript → NCNN…"
echo "  (This may take a minute — PNNX traces and optimizes the graph)…"

"${PNNX_TOOL}" "${TS_FILE}" \
    inputshape="[1,3,640,640]" \
    runtime=1 \
    optlevel=3 \
    2>&1 | tee "${OUTPUT_DIR}/pnnx.log" || true

# PNNX outputs files named after the input file stem, e.g.:
#   yolov10n.torchscript.ncnn.param
#   yolov10n.torchscript.ncnn.bin

# Find and rename to clean names
PARAM_FOUND=""
BIN_FOUND=""

for f in "${OUTPUT_DIR}/${MODEL}"*.ncnn.param; do
    if [[ -f "$f" ]]; then
        mv "$f" "${OUTPUT_DIR}/${MODEL}.ncnn.param"
        PARAM_FOUND="yes"
        echo "  ✓ Param: ${OUTPUT_DIR}/${MODEL}.ncnn.param"
    fi
done

for f in "${OUTPUT_DIR}/${MODEL}"*.ncnn.bin; do
    if [[ -f "$f" ]]; then
        mv "$f" "${OUTPUT_DIR}/${MODEL}.ncnn.bin"
        BIN_FOUND="yes"
        echo "  ✓ Bin:   ${OUTPUT_DIR}/${MODEL}.ncnn.bin"
    fi
done

# ── Verify output ────────────────────────────────────────────────────────────
if [[ -n "${PARAM_FOUND}" ]] && [[ -n "${BIN_FOUND}" ]]; then
    echo ""
    echo "  ✅ NCNN conversion complete."
    echo ""
    echo "  Model graph (first 30 lines):"
    head -30 "${OUTPUT_DIR}/${MODEL}.ncnn.param"
    echo "  …"
    echo ""
    echo "  Files:"
    ls -lh "${OUTPUT_DIR}/${MODEL}.ncnn."{param,bin}
else
    echo ""
    echo "  ⚠  PNNX output not found. Check the log:"
    echo "     ${OUTPUT_DIR}/pnnx.log"
    echo ""
    echo "  Common issues:"
    echo "  - Unsupported operators (TopK should work with the vlordier fork)"
    echo "  - Try running: grep -i 'ignore\\|todo\\|error' ${OUTPUT_DIR}/pnnx.log"
    echo ""
    echo "  Last 30 lines of PNNX log:"
    tail -30 "${OUTPUT_DIR}/pnnx.log"
    exit 1
fi
