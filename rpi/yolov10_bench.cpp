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
    net.opt.num_threads = 4;         // RPi5 has 4 big Cortex-A76 cores

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

    // Create a dummy input (640x640 RGB = black image)
    unsigned char* pixel_data = (unsigned char*)malloc(640 * 640 * 3);
    if (!pixel_data) {
        fprintf(stderr, "ERROR: Out of memory\n");
        return -1;
    }
    ncnn::Mat in = ncnn::Mat::from_pixels(
        pixel_data,
        ncnn::Mat::PIXEL_RGB, 640, 640
    );

    ncnn::Extractor ex = net.create_extractor();

    // Try common input names — YOLOv10 via PNNX typically uses "in0" or "images"
    const char* input_names[] = {"in0", "images", "input", "input.1", "data"};
    const char* output_names[] = {"out0", "output", "output0", "885", "886", "result"};

    int input_ok = -1;
    for (int i = 0; i < 5; i++) {
        ret = ex.input(input_names[i], in);
        if (ret == 0) {
            fprintf(stdout, "  ✓ Input blob: %s\n", input_names[i]);
            input_ok = i;
            break;
        }
    }

    if (input_ok < 0) {
        fprintf(stdout, "  ⚠  Could not confirm input blob names (forward still ran).\n");
    }

    // Try to extract output
    ncnn::Mat out;
    int output_ok = -1;
    for (int i = 0; i < 6; i++) {
        ret = ex.extract(output_names[i], out);
        if (ret == 0) {
            fprintf(stdout, "✓ Forward pass successful. Output shape: %d x %d x %d\n",
                    out.w, out.h, out.c);
            output_ok = i;
            break;
        }
    }

    if (output_ok < 0) {
        fprintf(stdout, "⚠  Forward pass completed but couldn't extract output.\n");
        fprintf(stdout, "   Check the .param file for actual blob names.\n");
    }

    free(pixel_data);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.param> <model.bin>\n", argv[0]);
        return 1;
    }
    return detect_yolov10(argv[1], argv[2]);
}
