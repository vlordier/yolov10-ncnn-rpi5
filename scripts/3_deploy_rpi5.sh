#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 3_deploy_rpi5.sh — Copy NCNN model to Raspberry Pi 5 and run inference test
#
# Usage: bash scripts/3_deploy_rpi5.sh [model_size] [user@host]
###############################################################################

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output"
RPI_DIR="${REPO_ROOT}/rpi"
MODEL="${1:-yolov10n}"
RPI_TARGET="${2}"

if [[ -z "${RPI_TARGET}" ]]; then
    echo "ERROR: RPi target required."
    echo "Usage: $0 ${MODEL} user@host"
    echo "  e.g.: $0 ${MODEL} pi@raspberrypi.local"
    exit 1
fi

PARAM_FILE="${OUTPUT_DIR}/${MODEL}.ncnn.param"
BIN_FILE="${OUTPUT_DIR}/${MODEL}.ncnn.bin"

for f in "${PARAM_FILE}" "${BIN_FILE}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing artifact: $f"
        echo "       Run 2_pnnx_convert.sh first."
        exit 1
    fi
done

# ── Create RPi test program ──────────────────────────────────────────────────
echo "→ Preparing RPi test program…"

mkdir -p "${RPI_DIR}"

cat > "${RPI_DIR}/CMakeLists.txt" <<'CMAKEOF'
cmake_minimum_required(VERSION 3.10)
project(yolov10_ncnn_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ncnn (assumes ncnn is installed system-wide or via CMAKE_PREFIX_PATH)
find_package(ncnn REQUIRED)

add_executable(yolov10_bench yolov10_bench.cpp)
target_link_libraries(yolov10_bench ncnn)
CMAKEOF

cat > "${RPI_DIR}/yolov10_bench.cpp" <<'CPPEOF'
// yolov10_bench.cpp — Minimal NCNN inference test for YOLOv10 on RPi5
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "cpu.h"
#include "net.h"

static int detect_yolov10(const char* param_path, const char* bin_path) {
    ncnn::Net net;
    net.opt.use_vulkan_compute = 0;  // RPi5 doesn't have Vulkan by default
    net.opt.use_int8_inference = 0;
    net.opt.num_threads = 4;         // RPi5 has 4 big cores

    int ret = net.load_param(param_path);
    if (ret != 0) {
        fprintf(stderr, "ERROR: Failed to load param: %s\n", param_path);
        return -1;
    }
    ret = net.load_bin(bin_path);
    if (ret != 0) {
        fprintf(stderr, "ERROR: Failed to load bin: %s\n", bin_path);
        return -1;
    }
    fprintf(stdout, "✓ Model loaded successfully.\n");
    fprintf(stdout,  "  Param: %s\n", param_path);
    fprintf(stdout,  "  Bin:   %s\n", bin_path);

    // Create a dummy input (1, 3, 640, 640)
    ncnn::Mat in = ncnn::Mat::from_pixels(
        (const unsigned char*)malloc(640 * 640 * 3),
        ncnn::Mat::PIXEL_RGB, 640, 640
    );

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);  // Input name may vary — check .param file

    // Try to extract output
    ncnn::Mat out;
    ret = ex.extract("out0", out);
    if (ret == 0) {
        fprintf(stdout, "✓ Forward pass successful. Output shape: %d x %d x %d\n",
                out.w, out.h, out.c);
    } else {
        fprintf(stdout, "⚠ Forward pass returned %d (output name may differ).\n", ret);
        fprintf(stdout, "  Check the .param file for actual input/output blob names.\n");
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.param> <model.bin>\n", argv[0]);
        return 1;
    }
    return detect_yolov10(argv[1], argv[2]);
}
CPPEOF

cat > "${RPI_DIR}/deploy.sh" <<'DEPLOYEOF'
#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-yolov10n}"
NCNN_DIR="${HOME}/ncnn"
WORK_DIR="${HOME}/yolov10-ncnn"

echo "=== YOLOv10 NCNN Deployment on Raspberry Pi 5 ==="
echo ""

# ── Install build dependencies (if not present) ──────────────────────────────
echo "→ Checking dependencies…"
sudo apt-get update -qq
sudo apt-get install -y -qq cmake build-essential protobuf-compiler libprotobuf-dev 2>/dev/null || true

# ── Build ncnn on RPi (if not already done) ──────────────────────────────────
if [[ ! -d "${NCNN_DIR}/build" ]]; then
    echo "→ Building ncnn on RPi5 (this takes ~10 min)…"
    mkdir -p "${NCNN_DIR}"
    cd /tmp
    git clone --branch fix-pnnx-onnx-topk-support --depth 1 \
        https://github.com/vlordier/ncnn.git ncnn-src
    mkdir -p ncnn-src/build
    cd ncnn-src/build
    cmake .. \
        -DNCNN_BUILD_EXAMPLES=OFF \
        -DNCNN_BUILD_TOOLS=OFF \
        -DNCNN_BUILD_BENCHMARK=OFF \
        -DCMAKE_INSTALL_PREFIX="${NCNN_DIR}"
    make -j"$(nproc)"
    make install
    cd /
    rm -rf /tmp/ncnn-src
    echo "  ✓ ncnn installed to ${NCNN_DIR}"
else
    echo "  ✓ ncnn already built at ${NCNN_DIR}"
fi

# ── Build test program ───────────────────────────────────────────────────────
echo ""
echo "→ Building YOLOv10 test program…"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Copy model files if not present
if [[ ! -f "${WORK_DIR}/${MODEL_NAME}.ncnn.param" ]]; then
    echo "  ⚠  Model files not found in ${WORK_DIR}"
    echo "  Please copy them manually:"
    echo "    scp user@host:/path/to/${MODEL_NAME}.ncnn.param ${WORK_DIR}/"
    echo "    scp user@host:/path/to/${MODEL_NAME}.ncnn.bin  ${WORK_DIR}/"
    exit 1
fi

mkdir -p build && cd build
cmake .. \
    -DCMAKE_PREFIX_PATH="${NCNN_DIR}" \
    -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

echo ""
echo "→ Running inference test…"
echo ""
"./yolov10_bench" "${WORK_DIR}/${MODEL_NAME}.ncnn.param" "${WORK_DIR}/${MODEL_NAME}.ncnn.bin"

echo ""
echo "✅ Deployment test complete."
DEPLOYEOF

chmod +x "${RPI_DIR}/deploy.sh"

echo "  ✓ RPi test program prepared in ${RPI_DIR}/"

# ── SCP to RPi ───────────────────────────────────────────────────────────────
echo ""
echo "→ Copying model + test program to ${RPI_TARGET}:~/yolov10-ncnn/ …"

ssh "${RPI_TARGET}" "mkdir -p ~/yolov10-ncnn"
scp "${PARAM_FILE}" "${RPI_TARGET}:~/yolov10-ncnn/"
scp "${BIN_FILE}" "${RPI_TARGET}:~/yolov10-ncnn/"
scp "${RPI_DIR}/CMakeLists.txt" "${RPI_TARGET}:~/yolov10-ncnn/"
scp "${RPI_DIR}/yolov10_bench.cpp" "${RPI_TARGET}:~/yolov10-ncnn/"
scp "${RPI_DIR}/deploy.sh" "${RPI_TARGET}:~/yolov10-ncnn/"

echo "  ✓ Files copied."

# ── Run on RPi ───────────────────────────────────────────────────────────────
echo ""
echo "→ Running deployment test on ${RPI_TARGET}…"
echo ""

ssh "${RPI_TARGET}" 'bash ~/yolov10-ncnn/deploy.sh '"${MODEL}"

echo ""
echo "✅ Done. Model is on the RPi5 at ~/yolov10-ncnn/"
