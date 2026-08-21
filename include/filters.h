#pragma once
#include <stdint.h>
#include <chrono>

// CPU Baseline Functinos
double cpu_gaussian_opencv(const char* filename);
double cpu_sharpen_opencv(const char* filename);
double cpu_edge_opencv(const char* filename);
double cpu_histogram_opencv(const char* filename);

// CUDA implementations
void launchGaussian(uint8_t* d_in, uint8_t* d_out, int width, int height);
void launchHistogram(uint8_t* d_in, int* d_hist, int n_pixels);
void launchEqualization(uint8_t* d_in, uint8_t* d_out, uint8_t* d_lut, int n_pixels);
void launchSharpen(uint8_t* d_in, uint8_t* d_out, int width, int height);
void launchEdgeDetect(uint8_t* d_in, uint8_t* d_out, int width, int height);
void launchSharpenEdgeFused(uint8_t* d_in, uint8_t* d_sharpened, uint8_t* d_edge, int width, int height);