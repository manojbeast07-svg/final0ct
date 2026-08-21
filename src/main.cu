#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb/stb_image.h"
#include "../include/stb/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "filters.h"

#define CHK(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    return 1; } } while (0)

int main (int argc, char *argv[])
{
    if(argc < 2){
        printf("Usage: %s <image>\n", argv[0]);
        return 1;
    }
    // GPU timing using CUDA events
    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));
    float gpu_ms;

    // check for CUDA devices
    int devCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&devCount);
    if(devErr != cudaSuccess || devCount == 0){
        fprintf(stderr, "No CUDA device found or CUDA not installed: %s\n", cudaGetErrorString(devErr));
        return 2;
    }

    // initialize host variables
    // load image
    

    printf("\nLoading image...\n");
    int width;
    int height;
    int channels;
    uint8_t* host_image_in = stbi_load(argv[1], &width, &height, &channels, 1);

    if (!host_image_in) {
        printf("Error: could not load image\n");
        return 1;
    }
    
    printf("Loaded: %d x %d\n", width, height);

    // allocate device and pinned host variables
    printf("Allocating device variables..."); fflush(stdout);
    printf("\n");
    uint8_t *A_in, *A_out_gaussian, *A_out_sharpen, *A_out_edge;

    size_t size = (size_t)width * (size_t)height;
    int n_pixels = width * height; // for histogram

    CHK(cudaMalloc((void**) &A_in, size));
    CHK(cudaMalloc((void**) &A_out_gaussian, size));
    CHK(cudaMalloc((void**) &A_out_sharpen, size));
    CHK(cudaMalloc((void**) &A_out_edge, size));

    int* d_hist;
    CHK(cudaMalloc((void**) &d_hist, 256 * sizeof(int)));
    CHK(cudaMemset(d_hist, 0, 256 * sizeof(int)));

    uint8_t* A_out_histogram;
    CHK(cudaMalloc((void**)&A_out_histogram, size));

    uint8_t* host_gaussian;
    uint8_t* host_histogram;
    uint8_t* host_sharpen;
    uint8_t* host_edge;
    uint8_t* host_image_pinned;
    CHK(cudaHostAlloc((void**)&host_image_pinned, size, cudaHostAllocDefault));
    CHK(cudaHostAlloc((void**)&host_gaussian, size, cudaHostAllocDefault));
    CHK(cudaHostAlloc((void**)&host_histogram, size, cudaHostAllocDefault));
    CHK(cudaHostAlloc((void**)&host_sharpen, size, cudaHostAllocDefault));
    CHK(cudaHostAlloc((void**)&host_edge, size, cudaHostAllocDefault));

    size_t total_device_bytes = size * 4 + 256 * sizeof(int) + 256 * sizeof(uint8_t);
    size_t total_host_pinned_bytes = size * 5;
    printf("[Memory] Device allocation: %zu bytes\n", total_device_bytes);
    printf("[Memory] Pinned host allocation: %zu bytes\n", total_host_pinned_bytes);

    CHK(cudaDeviceSynchronize());

    // copy image from CPU to GPU
    printf("Copying data from host to device..."); fflush(stdout);
    printf("\n");

    memcpy(host_image_pinned, host_image_in, size);
    stbi_image_free(host_image_in);
    CHK(cudaMemcpy(A_in, host_image_pinned, size, cudaMemcpyHostToDevice));
    CHK(cudaDeviceSynchronize());

    // launching kernel ----------------------------------------
    printf("Launching kernels..."); fflush(stdout);
    printf("\n");

    printf("Launching Gaussian kernel..."); fflush(stdout);
    printf("\n");
    // call Gaussian. function from filters.cu on this line
    launchGaussian(A_in, A_out_gaussian, width, height);
    CHK(cudaDeviceSynchronize());
    cudaEventRecord(start);
    launchGaussian(A_in, A_out_gaussian, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_ms, start, stop);
    float gpu_gaussian_ms = gpu_ms;
    if(cudaGetLastError() != cudaSuccess) fprintf(stderr, "Gaussian kernel launch error: %s\n", cudaGetErrorString(cudaGetLastError()));

    // CPU version
    double cpu_gaussian_ms = cpu_gaussian_opencv(argv[1]);
    printf("[Gaussian Blur]\n");
    printf("  GPU: %.4f ms\n", gpu_ms);
    printf("  CPU: %.4f ms\n", cpu_gaussian_ms);
    printf("  Speedup: %.2fx\n", cpu_gaussian_ms / gpu_ms);

    
    // pass gaussian blur image to histogram, noise is already removed
    // histogram. reflects true brightness distribution
    printf("Building histogram..."); fflush(stdout);
    printf("\n");
    cudaEventRecord(start);
    launchHistogram(A_out_gaussian, d_hist, n_pixels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_ms, start, stop);
    float gpu_histogram_ms = gpu_ms;
    if(cudaGetLastError() != cudaSuccess) fprintf(stderr, "Histogram kernel launch error: %s\n", cudaGetErrorString(cudaGetLastError()));

    // d_hist is GPU array of 256 integers
    double cpu_histogram_ms = cpu_histogram_opencv(argv[1]);

    // copy histogram to CPU
    // copy array back to CPU array
    int h_hist[256];
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // compute Cumulative Distribution Function running total
    int cdf[256];
    cdf[0] = h_hist[0];
    for(int i = 1; i < 256; i++){
        cdf[i] = cdf[i-1] + h_hist[i];
    }


    // tells us how many pixels have brighness less than or equal to the minimum
    // find first nonzero CDF value
    int cdf_min = 0;
    for(int i = 0; i < 256; i++){
        if(cdf[i] > 0){
            cdf_min = cdf[i];
            break;
        }
    }

    // build lookup table
    // for every possible brightness value 0-255, compute what new brightness should be
    uint8_t h_lut[256];
    for(int i = 0; i < 256; i++){
        h_lut[i] = n_pixels == cdf_min
            ? 0
            : (uint8_t)(((float)(cdf[i] - cdf_min) / (float)(n_pixels - cdf_min)) * 255.0f);
        // detail that was hidden in dark range becomes visible
    }

    // upload look up table to GPU
    // upload look up table to GPU so kernel can use it
    uint8_t* d_lut;
    CHK(cudaMalloc((void**)&d_lut, 256 * sizeof(uint8_t)));
    CHK(cudaMemcpy(d_lut, h_lut, 256 * sizeof(uint8_t), cudaMemcpyHostToDevice));

    printf("Applying Equalization..."); fflush(stdout);
    printf("\n");
    cudaEventRecord(start);
    launchEqualization(A_out_gaussian, A_out_histogram, d_lut, n_pixels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_histogram_ms += gpu_ms;
    if(cudaGetLastError() != cudaSuccess) fprintf(stderr, "Equalization kernel launch error: %s\n", cudaGetErrorString(cudaGetLastError()));

    cudaError_t cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel\n");
    printf("\n");

    printf("[Histogram Equalization]\n");
    printf("  GPU: %.4f ms\n", gpu_histogram_ms);
    printf("  CPU: %.4f ms\n", cpu_histogram_ms);
    printf("  Speedup: %.2fx\n", cpu_histogram_ms / gpu_histogram_ms);

    // Fuse small-image Sharpen and Edge to remove one launch and an intermediate
    // global-memory round trip. Large images retain the tiled two-kernel path.
    printf("\nLaunching Sharpen/Edge kernels...\n"); fflush(stdout);
    float gpu_sharpen_ms = 0.0f;
    float gpu_edge_ms = 0.0f;
    cudaEventRecord(start);
    if ((size_t)width * (size_t)height < 2u * 1024u * 1024u) {
        launchSharpenEdgeFused(A_out_histogram, A_out_sharpen, A_out_edge, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_ms, start, stop);
        gpu_sharpen_ms = gpu_ms;
    } else {
        launchSharpen(A_out_histogram, A_out_sharpen, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_sharpen_ms, start, stop);
        cudaEventRecord(start);
        launchEdgeDetect(A_out_sharpen, A_out_edge, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_edge_ms, start, stop);
    }
    gpu_ms = gpu_sharpen_ms;
    if(cudaGetLastError() != cudaSuccess) fprintf(stderr, "Sharpen kernel launch error: %s\n", cudaGetErrorString(cudaGetLastError()));

    // cpu version
    double cpu_sharpen_ms = cpu_sharpen_opencv(argv[1]);
    printf("[Sharpen%s]\n", ((size_t)width * (size_t)height < 2u * 1024u * 1024u) ? " + Edge fused" : "");
    printf("  GPU: %.4f ms\n", gpu_ms);
    printf("  CPU: %.4f ms\n", cpu_sharpen_ms);
    printf("  Speedup: %.2fx\n", cpu_sharpen_ms / gpu_ms);

    if(cudaGetLastError() != cudaSuccess) fprintf(stderr, "EdgeDetect kernel launch error: %s\n", cudaGetErrorString(cudaGetLastError()));

    // cpu version
    double cpu_edge_ms = cpu_edge_opencv(argv[1]);
    printf("[Edge Detection%s]\n", ((size_t)width * (size_t)height < 2u * 1024u * 1024u) ? " fused above" : "");
    printf("  GPU: %.4f ms\n", gpu_edge_ms);
    printf("  CPU: %.4f ms\n", cpu_edge_ms);
    printf("  Speedup: %.2fx\n", gpu_edge_ms > 0.0f ? cpu_edge_ms / gpu_edge_ms : 0.0f);

    
    cuda_ret = cudaDeviceSynchronize();

    // Copy device variables to host and save------------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    printf("\n");

    // Save results
    cudaMemcpy(host_gaussian, A_out_gaussian, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_histogram, A_out_histogram, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_sharpen, A_out_sharpen, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_edge, A_out_edge, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    stbi_write_png("output/gaussian.png",   width, height, 1, host_gaussian,  width);
    printf("Saved output/gaussian.png\n");
    printf("\n");

    stbi_write_png("output/histogram.png",  width, height, 1, host_histogram, width);
    printf("Saved output/histogram.png\n");
    printf("\n");

    stbi_write_png("output/sharpen.png", width, height, 1, host_sharpen, width);
    printf("Saved output/sharpen.png\n");
    printf("\n");

    stbi_write_png("output/edge.png", width, height, 1, host_edge, width);
    printf("Saved output/edge.png\n");
    printf("\n");
    
    // ── Combined preview image ────────────────────────────────────────────────
    // gaussian(top-left) | histogram(top-right)
    // sharpen(bottom-left) | edge(bottom-right)
    int combined_width  = width * 2;
    int combined_height = height * 2;
    uint8_t* combined = (uint8_t*)calloc(combined_width * combined_height, 1);

    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            int src = y * width + x;
            combined[y * combined_width + x]                      = host_gaussian[src];
            combined[y * combined_width + (x + width)]            = host_histogram[src];
            combined[(y + height) * combined_width + x]           = host_sharpen[src];
            combined[(y + height) * combined_width + (x + width)] = host_edge[src];
        }
    }
    stbi_write_png("output/preview.png", combined_width, combined_height, 1, combined, combined_width);

    // cleanup all allocated resources
    free(combined);
    CHK(cudaFree(A_in));
    CHK(cudaFree(A_out_gaussian));
    CHK(cudaFree(A_out_sharpen));
    CHK(cudaFree(A_out_edge));
    CHK(cudaFree(A_out_histogram));
    CHK(cudaFree(d_hist));
    CHK(cudaFreeHost(host_image_pinned));
    CHK(cudaFreeHost(host_gaussian));
    CHK(cudaFreeHost(host_histogram));
    CHK(cudaFreeHost(host_sharpen));
    CHK(cudaFreeHost(host_edge));
    CHK(cudaFree(d_lut));
    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));

    printf("{\"gpu\": {\"gaussian_ms\": %.4f, \"histogram_ms\": %.4f, \"sharpen_ms\": %.4f, \"edge_ms\": %.4f},\n", gpu_gaussian_ms, gpu_histogram_ms, gpu_sharpen_ms, gpu_edge_ms);
    printf(" \"cpu\": {\"gaussian_ms\": %.4f, \"histogram_ms\": %.4f, \"sharpen_ms\": %.4f, \"edge_ms\": %.4f}\n}\n", cpu_gaussian_ms, cpu_histogram_ms, cpu_sharpen_ms, cpu_edge_ms);

    return 0;
}