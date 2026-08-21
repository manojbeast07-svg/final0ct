#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb/stb_image.h"
#include "../include/stb/stb_image_write.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call) do { cudaError_t error = (call); if (error != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); return 1; } } while (0)

__device__ __forceinline__ uint8_t pixel(const uint8_t* image, int x, int y, int width, int height) {
    x = max(0, min(width - 1, x));
    y = max(0, min(height - 1, y));
    return image[y * width + x];
}

__global__ void gaussian_simple(const uint8_t* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int sum = pixel(input, x - 1, y - 1, width, height) + pixel(input, x + 1, y - 1, width, height)
        + pixel(input, x - 1, y + 1, width, height) + pixel(input, x + 1, y + 1, width, height)
        + 2 * (pixel(input, x, y - 1, width, height) + pixel(input, x - 1, y, width, height)
        + pixel(input, x + 1, y, width, height) + pixel(input, x, y + 1, width, height))
        + 4 * pixel(input, x, y, width, height);
    output[y * width + x] = (uint8_t)((sum + 8) >> 4);
}

__global__ void histogram_simple(const uint8_t* input, unsigned int* histogram, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) atomicAdd(&histogram[input[index]], 1);
}

__global__ void equalize_simple(const uint8_t* input, uint8_t* output, const uint8_t* lut, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = lut[input[index]];
}

__global__ void sharpen_simple(const uint8_t* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float sum = -0.25f * (pixel(input, x - 1, y - 1, width, height) + pixel(input, x + 1, y - 1, width, height)
        + pixel(input, x - 1, y + 1, width, height) + pixel(input, x + 1, y + 1, width, height))
        - 0.5f * (pixel(input, x, y - 1, width, height) + pixel(input, x - 1, y, width, height)
        + pixel(input, x + 1, y, width, height) + pixel(input, x, y + 1, width, height))
        + 4.0f * pixel(input, x, y, width, height);
    output[y * width + x] = (uint8_t)max(0.0f, min(255.0f, roundf(sum)));
}

__global__ void edge_simple(const uint8_t* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int value = pixel(input, x - 1, y - 1, width, height) - pixel(input, x + 1, y - 1, width, height)
        - pixel(input, x - 1, y + 1, width, height) + pixel(input, x + 1, y + 1, width, height);
    output[y * width + x] = (uint8_t)max(0, min(255, value));
}

static void save_stage(const char* name, int width, int height, const uint8_t* data) {
    char path[256];
    snprintf(path, sizeof(path), "output/gpu1_%s.png", name);
    stbi_write_png(path, width, height, 1, data, width);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <image>\n", argv[0]); return 1; }
    int width, height, channels;
    uint8_t* host_input = stbi_load(argv[1], &width, &height, &channels, 1);
    if (!host_input) { fprintf(stderr, "Could not load image\n"); return 1; }
    int count = width * height;
    size_t bytes = (size_t)count;
    printf("[Memory] Device allocation: %zu bytes\n", bytes * 5 + 256 * sizeof(unsigned int) + 256);
    uint8_t *input, *gaussian, *equalized, *sharpened, *edge;
    unsigned int* histogram;
    uint8_t* lut;
    CUDA_CHECK(cudaMalloc(&input, bytes));
    CUDA_CHECK(cudaMalloc(&gaussian, bytes));
    CUDA_CHECK(cudaMalloc(&equalized, bytes));
    CUDA_CHECK(cudaMalloc(&sharpened, bytes));
    CUDA_CHECK(cudaMalloc(&edge, bytes));
    CUDA_CHECK(cudaMalloc(&histogram, 256 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&lut, 256));
    CUDA_CHECK(cudaMemcpy(input, host_input, bytes, cudaMemcpyHostToDevice));
    stbi_image_free(host_input);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    int linear_blocks = (count + 255) / 256;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsed;

    gaussian_simple<<<blocks, threads>>>(input, gaussian, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(start);
    gaussian_simple<<<blocks, threads>>>(input, gaussian, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("[GPU1 Gaussian Blur] GPU: %.4f ms\n", elapsed);
    uint8_t* host_stage = (uint8_t*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(host_stage, gaussian, bytes, cudaMemcpyDeviceToHost)); save_stage("gaussian", width, height, host_stage);

    CUDA_CHECK(cudaMemset(histogram, 0, 256 * sizeof(unsigned int)));
    cudaEventRecord(start); histogram_simple<<<linear_blocks, 256>>>(gaussian, histogram, count); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); float histogram_kernel_ms = elapsed;
    unsigned int host_histogram[256]; CUDA_CHECK(cudaMemcpy(host_histogram, histogram, sizeof(host_histogram), cudaMemcpyDeviceToHost));
    int cumulative = 0, minimum = 0;
    for (int i = 0; i < 256; ++i) {
        cumulative += host_histogram[i];
        if (cumulative > 0) {
            minimum = cumulative;
            break;
        }
    }
    cumulative = 0; uint8_t host_lut[256]; for (int i = 0; i < 256; ++i) { cumulative += host_histogram[i]; host_lut[i] = (uint8_t)(count == minimum ? 0 : ((float)(cumulative - minimum) / (count - minimum)) * 255.0f); }
    CUDA_CHECK(cudaMemcpy(lut, host_lut, 256, cudaMemcpyHostToDevice));
    cudaEventRecord(start); equalize_simple<<<linear_blocks, 256>>>(gaussian, equalized, lut, count); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); printf("[GPU1 Histogram Equalization] GPU: %.4f ms\n", histogram_kernel_ms + elapsed); CUDA_CHECK(cudaMemcpy(host_stage, equalized, bytes, cudaMemcpyDeviceToHost)); save_stage("histogram", width, height, host_stage);

    cudaEventRecord(start); sharpen_simple<<<blocks, threads>>>(equalized, sharpened, width, height); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); printf("[GPU1 Sharpen] GPU: %.4f ms\n", elapsed); CUDA_CHECK(cudaMemcpy(host_stage, sharpened, bytes, cudaMemcpyDeviceToHost)); save_stage("sharpen", width, height, host_stage);
    cudaEventRecord(start); edge_simple<<<blocks, threads>>>(sharpened, edge, width, height); cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&elapsed, start, stop); printf("[GPU1 Edge Detection] GPU: %.4f ms\n", elapsed); CUDA_CHECK(cudaMemcpy(host_stage, edge, bytes, cudaMemcpyDeviceToHost)); save_stage("edge", width, height, host_stage);

    free(host_stage); cudaFree(input); cudaFree(gaussian); cudaFree(equalized); cudaFree(sharpened); cudaFree(edge); cudaFree(histogram); cudaFree(lut); cudaEventDestroy(start); cudaEventDestroy(stop); return 0;
}
