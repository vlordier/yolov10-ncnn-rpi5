# YOLOv10 → NCNN for Raspberry Pi 5

Converts a YOLOv10 model to NCNN format (`.param` + `.bin`) using the **vlordier/ncnn** fork that adds TopK operator support — the missing piece that previously blocked YOLOv10 deployment on NCNN/Raspberry Pi.

## Prerequisites

- **Host machine** (x86_64 or aarch64) with Python ≥ 3.8
- **Raspberry Pi 5** (aarch64 / ARM64) — target deployment device
- Git, CMake ≥ 3.10, a C++ compiler
- ~4 GB free disk space

## Quick Start

```bash
# 1. Clone this repo
git clone <your-repo-url> yolov10-ncnn-rpi5
cd yolov10-ncnn-rpi5

# 2. Run the full conversion pipeline (outputs to ./output/)
./run.sh

# 3. Deploy to your RPi5 (optional — requires SSH access)
./run.sh yolov10n pi@raspberrypi.local
```

`run.sh` orchestrates the three stages below. You can also run each stage individually.

## Pipeline Stages

| Stage | Script | What it does |
|-------|--------|-------------|
| lint | `scripts/lint.sh` | Run shellcheck on all shell scripts |
| 0 | `scripts/0_setup.sh` | Clone & build vlordier/ncnn (with TopK), install ultralytics |
| 1 | `scripts/1_export_onnx.sh` | Download YOLOv10n from Ultralytics → export to ONNX |
| 2 | `scripts/2_pnnx_convert.sh` | PyTorch → TorchScript → PNNX → NCNN (`.param` + `.bin`) |
| 3 | `scripts/3_deploy_rpi5.sh` | Copy artifacts to RPi5 and run a quick inference test |

## Output

After a successful run you'll find:

```
output/
├── yolov10n.torchscript.pt       # TorchScript intermediate
├── yolov10n.onnx                  # ONNX export (optional)
├── yolov10n.torchscript.pnnx.*    # PNNX debug format
├── yolov10n.ncnn.param            # NCNN model graph (with TopK!)
└── yolov10n.ncnn.bin              # NCNN model weights
```

## Why This Works

YOLOv10 integrates NMS (including a **TopK** operator) directly into the model graph.
The upstream NCNN/Tencent PNNX pipeline historically marked TopK as `ignore` / `todo`,
making YOLOv10 unusable on NCNN.

This repo patches the **vlordier/ncnn** fork (`fix-pnnx-onnx-topk-support` branch) to fix
the pattern matching between pass_level2 and pass_ncnn:

1. **`pass_level2/torch_topk.cpp`** — captures `k`, `dim`, `largest`, `sorted` as **parameters**
   (not tensor inputs) so they can be passed to the ncnn layer
2. **`pass_ncnn/TopK.cpp`** — matches the resulting `torch.topk` operator and converts it to
   an ncnn `TopK` layer with proper axis/largest/sorted parameters

### Before fix:
```
ignore torch.topk torch.topk_11 param dim=-1
ignore torch.topk torch.topk_11 param k=300
```

### After fix:
```
TopK                     topk_162   1 2 296 297 298 0=-1 1=1 2=1
TopK                     topk_163   1 2 304 305 306 0=-1 1=1 2=1
```

## Troubleshooting

### `pnnx: command not found`
PNNX is built as part of the ncnn fork and placed in `ncnn/build/tools/pnnx`. The scripts add it to `$PATH`.

### TopK still fails on a larger model (e.g. yolov10s/m/l)
The TopK implementation may have limitations with dynamic axes or very large K values. Start with `yolov10n` (nano) which has the smallest post-processing graph.

### RPi5 inference is slow
YOLOv10 with full NMS inside the graph is heavy for the Pi. Consider:
- Using `yolov10n` (nano variant)
- INT8 quantization via `ncnn2table` + `ncnn2int8`
- Moving post-processing out of the graph (export with `agnostic=True, retina_masks=False`)

## References

- [Ultralytics YOLOv10](https://github.com/THU-MIG/yolov10)
- [vlordier/ncnn — TopK branch](https://github.com/vlordier/ncnn/tree/fix-pnnx-onnx-topk-support)
- [NCNN wiki — PyTorch/ONNX to NCNN](https://github.com/tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx)
- [NCNN TopK feature request](https://github.com/Tencent/ncnn/issues/6377)
