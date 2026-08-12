#include <stdio.h>
#include "../include/filters.h"
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace cv;
using namespace std::chrono;

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8
// total threads per block = 256
// 32 x 8 thread block for good x-direction memory coalescing

static __device__ __forceinline__ int clamp_coord(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static __device__ __forceinline__ uint8_t read_pixel(const uint8_t* __restrict__ in, int x, int y, int width, int height) {
    x = clamp_coord(x, 0, width - 1);
    y = clamp_coord(y, 0, height - 1);
    return __ldg(&in[y * width + x]);
}

__constant__ float GAUSSIAN_MASK[3][3] = {
    {1/16.f, 2/16.f, 1/16.f},
    {2/16.f, 4/16.f, 2/16.f},
    {1/16.f, 2/16.f, 1/16.f}
};

__constant__ float SHARPEN_MASK[3][3] = {
    {-0.25f, -0.5f, -0.25f},
    {-0.5f, 4.0f, -0.5f},
    {-0.25f, -0.5f, -0.25f}
};

__constant__ int SOBEL_X[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int SOBEL_Y[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}
};

// ------------------------------------------
// KERNELS
// ------------------------------------------
/*
every pixel replace it with a weight average of itself and its neighbors
2D thread grid - one thread per pixel

Each thread computes 3x3 convolution using the mask

Neighbor coordinates clamped with min/max to handle the borders of the image (out of bounds)

Block size same for all kernels

use opencv for cpu baseline
use my own pipeline for gpu comparison

*/
__global__ void gaussian(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int width, int height) {
    __shared__ uint8_t tile[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int x = bx + tx;
    int y = by + ty;
    int base_x = bx - 1;
    int base_y = by - 1;
    bool block_interior = (bx > 0 && bx + BLOCK_DIM_X + 1 < width && by > 0 && by + BLOCK_DIM_Y + 1 < height);

    for (int row = ty; row < BLOCK_DIM_Y + 2; row += blockDim.y) {
        for (int col = tx; col < BLOCK_DIM_X + 2; col += blockDim.x) {
            int sx = base_x + col;
            int sy = base_y + row;
            if (block_interior) {
                tile[row][col] = __ldg(&in[sy * width + sx]);
            } else {
                tile[row][col] = read_pixel(in, sx, sy, width, height);
            }
        }
    }
    __syncthreads();

    if (x >= width || y >= height) return;

    int sum = 0;
    sum += tile[ty + 0][tx + 0];
    sum += tile[ty + 0][tx + 2];
    sum += tile[ty + 2][tx + 0];
    sum += tile[ty + 2][tx + 2];
    sum += 2 * tile[ty + 0][tx + 1];
    sum += 2 * tile[ty + 1][tx + 0];
    sum += 2 * tile[ty + 1][tx + 2];
    sum += 2 * tile[ty + 2][tx + 1];
    sum += 4 * tile[ty + 1][tx + 1];

    out[y * width + x] = (uint8_t)((sum + 8) >> 4);
}

__global__ void gaussian_direct(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int sum = read_pixel(in, x - 1, y - 1, width, height) + read_pixel(in, x + 1, y - 1, width, height)
        + read_pixel(in, x - 1, y + 1, width, height) + read_pixel(in, x + 1, y + 1, width, height)
        + 2 * (read_pixel(in, x, y - 1, width, height) + read_pixel(in, x - 1, y, width, height)
        + read_pixel(in, x + 1, y, width, height) + read_pixel(in, x, y + 1, width, height))
        + 4 * read_pixel(in, x, y, width, height);
    out[y * width + x] = (uint8_t)((sum + 8) >> 4);
}

/*
Know distribution of how many pixels are dark and how many are bright

count how many pixels have each brightness value and put them into each bucket
use privitazation from technique , each thread block gets its own private histogram in SM
atomic operations instead of global memory operations
merge private histograms using atomicAdd
*/
__global__ void buildHistogram(uint8_t* in, int* histogram, int n_pixels){
    __shared__ unsigned int histo_private[256]; // 256 bins fixed size

    for(int i = threadIdx.x; i < 256; i += blockDim.x) {
        histo_private[i] = 0;
    }
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while(i < n_pixels){
        atomicAdd(&histo_private[in[i]], 1);
        i += stride;
    }
    __syncthreads();

    for(int j = threadIdx.x; j < 256; j += blockDim.x) {
        atomicAdd(&(histogram[j]), histo_private[j]);
    }
}

/*
    Applying the histogram to the image and change the pixels
    each thread reads one pixel, looks up the new value in the lookup table, writes it to the image
*/
__global__ void applyEqualization(uint8_t* in, uint8_t* out, uint8_t* lut, int n_pixels){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= n_pixels) return;

    out[idx] = lut[in[idx]]; 
}

/*
Enhance fine detail by applying a direct sharpening convolution mask.
*/
__global__ void sharpen_direct(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int width, int height){
    __shared__ uint8_t tile[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int x = bx + tx;
    int y = by + ty;
    int base_x = bx - 1;
    int base_y = by - 1;
    bool block_interior = (bx > 0 && bx + BLOCK_DIM_X + 1 < width && by > 0 && by + BLOCK_DIM_Y + 1 < height);

    for (int row = ty; row < BLOCK_DIM_Y + 2; row += blockDim.y) {
        for (int col = tx; col < BLOCK_DIM_X + 2; col += blockDim.x) {
            int sx = base_x + col;
            int sy = base_y + row;
            if (block_interior) {
                tile[row][col] = __ldg(&in[sy * width + sx]);
            } else {
                tile[row][col] = read_pixel(in, sx, sy, width, height);
            }
        }
    }
    __syncthreads();

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    sum += SHARPEN_MASK[0][0] * tile[ty + 0][tx + 0];
    sum += SHARPEN_MASK[0][1] * tile[ty + 0][tx + 1];
    sum += SHARPEN_MASK[0][2] * tile[ty + 0][tx + 2];
    sum += SHARPEN_MASK[1][0] * tile[ty + 1][tx + 0];
    sum += SHARPEN_MASK[1][1] * tile[ty + 1][tx + 1];
    sum += SHARPEN_MASK[1][2] * tile[ty + 1][tx + 2];
    sum += SHARPEN_MASK[2][0] * tile[ty + 2][tx + 0];
    sum += SHARPEN_MASK[2][1] * tile[ty + 2][tx + 1];
    sum += SHARPEN_MASK[2][2] * tile[ty + 2][tx + 2];

    int result = (int)roundf(sum);
    out[y * width + x] = (uint8_t)min(max(result, 0), 255);
}


/*
calculate how fast pixel values change across the image. 
detect bone edge and brighten it, darken soft tissue on the other. 

fractures will be more apparent now
measures how fast brightness changes at every pixel
bone boundaries are where brightness changes quickly, highlights edges

Same thread initialization, loop over same but with 2 different masks
combine the two sums with sqrtf()

bound the results to prevent boundary edge detection
*/
__global__ void edgeDetect(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int width, int height){
    __shared__ uint8_t tile[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int x = bx + tx;
    int y = by + ty;
    int base_x = bx - 1;
    int base_y = by - 1;
    bool block_interior = (bx > 0 && bx + BLOCK_DIM_X + 1 < width && by > 0 && by + BLOCK_DIM_Y + 1 < height);

    for (int row = ty; row < BLOCK_DIM_Y + 2; row += blockDim.y) {
        for (int col = tx; col < BLOCK_DIM_X + 2; col += blockDim.x) {
            int sx = base_x + col;
            int sy = base_y + row;
            if (block_interior) {
                tile[row][col] = __ldg(&in[sy * width + sx]);
            } else {
                tile[row][col] = read_pixel(in, sx, sy, width, height);
            }
        }
    }
    __syncthreads();

    if (x >= width || y >= height) return;

    // Match OpenCV Sobel with dx=1, dy=1, ksize=3.
    // This is the mixed derivative kernel used by the CPU baseline.
    int sum = 0;
    sum += tile[ty + 0][tx + 0];
    sum += -tile[ty + 0][tx + 2];
    sum += -tile[ty + 2][tx + 0];
    sum += tile[ty + 2][tx + 2];

    out[y * width + x] = (uint8_t)min(max(sum, 0), 255);
}


// ------------------------------------------
// KERNEL CALLS
// ------------------------------------------
void launchGaussian(uint8_t* d_in, uint8_t* d_out, int width, int height)
{
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 blocks(
        ((width + BLOCK_DIM_X - 1) / BLOCK_DIM_X),
        ((height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y)
    );
    if ((size_t)width * (size_t)height < 2u * 1024u * 1024u) {
        dim3 direct_threads(16, 16);
        dim3 direct_blocks((width + 15) / 16, (height + 15) / 16);
        gaussian_direct<<<direct_blocks, direct_threads>>>(d_in, d_out, width, height);
    } else {
        gaussian<<<blocks, threads>>>(d_in, d_out, width, height);
    }
}


void launchHistogram(uint8_t* d_in, int* d_hist, int n_pixels){
    int threads = 256;
    int blocks = (n_pixels + threads - 1) / threads;
    buildHistogram<<<blocks, threads>>>(d_in, d_hist, n_pixels);
}

void launchEqualization(uint8_t* d_in, uint8_t* d_out, uint8_t* d_lut, int n_pixels){
    int threads = 256;
    int blocks = (n_pixels + threads - 1) / threads;
    applyEqualization<<<blocks, threads>>>(d_in, d_out, d_lut, n_pixels);
}


void launchSharpen(uint8_t* d_in, uint8_t* d_out, int width, int height)
{
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 blocks(
        ((width + BLOCK_DIM_X - 1) / BLOCK_DIM_X),
        ((height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y)
    );
    sharpen_direct<<<blocks, threads>>>(d_in, d_out, width, height);
}

void launchEdgeDetect(uint8_t* d_in, uint8_t* d_out, int width, int height)
{
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 blocks(
        ((width + BLOCK_DIM_X - 1) / BLOCK_DIM_X),
        ((height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y)
    );
    edgeDetect<<<blocks, threads>>>(d_in, d_out, width, height);
}

// CPU Baseline Filter Functions:
// call gaussian blur function from opencv
double cpu_gaussian_opencv(const char* filename) {
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    if (img.empty()) {
        return 0.0;
    }
    Mat out;
    auto start = high_resolution_clock::now();
    GaussianBlur(img, out, Size(3, 3), 0.85);
    auto end = high_resolution_clock::now();
    return duration<double, std::milli>(end - start).count();
}

// call Gaussian blur just like Kernel, but then different weight
double cpu_sharpen_opencv(const char* filename) {
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    if (img.empty()) {
        return 0.0;
    }
    Mat blurred, equalized, out;
    auto start = high_resolution_clock::now();
    GaussianBlur(img, blurred, Size(3, 3), 0.85);
    equalizeHist(blurred, equalized);
    Mat kernel = (Mat_<float>(3, 3) << -0.25f, -0.5f, -0.25f,
                                      -0.5f,  4.0f, -0.5f,
                                      -0.25f, -0.5f, -0.25f);
    filter2D(equalized, out, CV_8U, kernel);
    auto end = high_resolution_clock::now();
    return duration<double, std::milli>(end - start).count();
}

// call sobel edge function from opencv
double cpu_edge_opencv(const char* filename) {
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    if (img.empty()) {
        return 0.0;
    }
    Mat blurred, equalized, sharpened, out;
    auto start = high_resolution_clock::now();
    GaussianBlur(img, blurred, Size(3, 3), 0.85);
    equalizeHist(blurred, equalized);
    Mat kernel = (Mat_<float>(3, 3) << -0.25f, -0.5f, -0.25f,
                                      -0.5f,  4.0f, -0.5f,
                                      -0.25f, -0.5f, -0.25f);
    filter2D(equalized, sharpened, CV_8U, kernel);
    Sobel(sharpened, out, CV_8U, 1, 1, 3);
    auto end = high_resolution_clock::now();
    return duration<double, std::milli>(end - start).count();
}

//call histogram function 
double cpu_histogram_opencv(const char* filename) {
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    if (img.empty()) {
        return 0.0;
    }
    Mat blurred, out;
    auto start = high_resolution_clock::now();
    GaussianBlur(img, blurred, Size(3, 3), 0.85);
    equalizeHist(blurred, out);
    auto end = high_resolution_clock::now();
    return duration<double, std::milli>(end - start).count();
}