#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 1_export_onnx.sh — Export YOLOv10 from Ultralytics to ONNX
#
# Usage: bash scripts/1_export_onnx.sh [model_size]
#   model_size: yolov10n (default), yolov10s, yolov10m, yolov10l, yolov10x
###############################################################################

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output"
MODEL="${1:-yolov10n}"

mkdir -p "${OUTPUT_DIR}"

echo "→ Exporting ${MODEL} to ONNX…"
echo "  (This will auto-download weights on first run.)"

python3 -c "
from pathlib import Path
from ultralytics import YOLO

model = YOLO('${MODEL}.pt')
out = model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    opset=17,
    dynamic=False,
    nms=True,
)
# Move to output dir
out_path = Path(out)
target = Path('${OUTPUT_DIR}') / '${MODEL}.onnx'
if out_path.resolve() != target.resolve():
    out_path.rename(target)
    print(f'  Moved {out_path.name} → ${OUTPUT_DIR}/')
print(f'  ✓ ONNX export complete: {target}')
"

if [[ -f "${OUTPUT_DIR}/${MODEL}.onnx" ]]; then
    echo "  ✓ Output: ${OUTPUT_DIR}/${MODEL}.onnx"
else
    echo "  ⚠  ONNX file not found in output dir. Check export logs."
    exit 1
fi
