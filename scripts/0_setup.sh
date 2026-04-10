#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0_setup.sh — Clone & build the vlordier/ncnn fork with TopK support
#
# PNNX is a separate CMake project under tools/pnnx/ that requires PyTorch.
# The main ncnn lib is built first, then PNNX is built independently.
###############################################################################

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NCNN_DIR="${REPO_ROOT}/ncnn"
NCNN_URL="https://github.com/vlordier/ncnn.git"
NCNN_BRANCH="fix-pnnx-onnx-topk-support"

NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

echo "→ Checking for required build tools…"
for cmd in git cmake; do
    command -v "${cmd}" &>/dev/null || { echo "ERROR: ${cmd} not found. Install it first."; exit 1; }
done

# Verify PyTorch (required for PNNX)
echo "  Checking PyTorch…"
python3 -c "
import torch
print(f'  ✓ PyTorch {torch.__version__}')
print(f'    cmake path: {torch.utils.cmake_prefix_path}')
" || { echo "ERROR: PyTorch not found. pip install torch first."; exit 1; }

# ── Clone ncnn fork ──────────────────────────────────────────────────────────
if [[ -d "${NCNN_DIR}/.git" ]]; then
    echo "  ✓ ncnn fork already cloned at ${NCNN_DIR}"
else
    echo "  Cloning ${NCNN_URL} → ${NCNN_DIR} (branch: ${NCNN_BRANCH})…"
    git clone --branch "${NCNN_BRANCH}" --depth 1 "${NCNN_URL}" "${NCNN_DIR}"
fi

# ── Build ncnn library ───────────────────────────────────────────────────────
if [[ -f "${NCNN_DIR}/build/src/libncnn.a" ]]; then
    echo "  ✓ ncnn lib already built — skipping."
else
    echo "  Building ncnn library…"
    mkdir -p "${NCNN_DIR}/build"
    cd "${NCNN_DIR}/build"
    cmake .. \
        -DNCNN_BUILD_EXAMPLES=OFF \
        -DNCNN_BUILD_TOOLS=ON \
        -DNCNN_BUILD_BENCHMARK=OFF
    make -j"${NCPU}"
    cd "${REPO_ROOT}"
    echo "  ✓ ncnn library built."
fi

# ── Build PNNX tool ─────────────────────────────────────────────────────────
if [[ -f "${NCNN_DIR}/tools/pnnx/build/src/pnnx" ]]; then
    echo "  ✓ PNNX tool already exists — skipping build."
else
    echo "  Building PNNX…"
    # Discover Torch cmake path from the correct Python
    TORCH_CMAKE_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    PYTHON3_EXE="$(command -v python3)"

    mkdir -p "${NCNN_DIR}/tools/pnnx/build"
    cd "${NCNN_DIR}/tools/pnnx/build"
    cmake .. \
        -DTorch_DIR="${TORCH_CMAKE_PATH}/Torch" \
        -DPython3_EXECUTABLE="${PYTHON3_EXE}" \
        -Donnxruntime_INSTALL_DIR=/opt/homebrew
    make -j"${NCPU}"
    cd "${REPO_ROOT}"
fi

# Verify PNNX exists
PNNX_BIN=""
if [[ -f "${NCNN_DIR}/tools/pnnx/build/src/pnnx" ]]; then
    PNNX_BIN="${NCNN_DIR}/tools/pnnx/build/src/pnnx"
elif [[ -f "${NCNN_DIR}/tools/pnnx/build/pnnx" ]]; then
    PNNX_BIN="${NCNN_DIR}/tools/pnnx/build/pnnx"
elif [[ -f "${NCNN_DIR}/tools/pnnx/build/src/pnnx.exe" ]]; then
    PNNX_BIN="${NCNN_DIR}/tools/pnnx/build/src/pnnx.exe"
fi

if [[ -n "${PNNX_BIN}" ]]; then
    echo "  ✓ PNNX: ${PNNX_BIN}"
else
    echo "  ⚠  PNNX binary not found."
    echo "     Check the build output above for errors."
    echo "     Look in: ${NCNN_DIR}/tools/pnnx/build/"
    exit 1
fi

# ── Install ultralytics (Python) ─────────────────────────────────────────────
echo "  Checking ultralytics package…"
if python3 -c "import ultralytics; print(f'  ✓ ultralytics {ultralytics.__version__}')" 2>/dev/null; then
    : # already installed
else
    echo "  Installing ultralytics…"
    pip3 install ultralytics --quiet
fi

echo ""
echo "✅ Setup complete."
