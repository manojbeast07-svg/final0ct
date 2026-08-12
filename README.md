<<<<<<< HEAD
# usecase11
=======
# ClearScan

Pipeline that takes raw X-ray image and produced enhanced version where bones are sharper, edges are more defined, and abnormalities are clear. 

The pipeline consists of:

1. Gaussian Blur
2. Histogram Equalization
3. Sharpening
4. Sobel Edge Detection

Each stage is implemented as a CUDA kernel and compared against an OpenCV CPU baseline.

Developed for CS/EE 147: GPU Programming at UC Riverside


# Demo Video

https://youtu.be/8CjhCsuPkCM

## Processing Pipeline

Input X-ray
     │
     ▼
Gaussian Blur
     │
     ▼
Histogram Equalization
     │
     ▼
Sharpening
     │
     ▼
Sobel Edge Detection
     │
     ▼
Enhanced Output

## Project Structure
```text
final-project-clearscan/
│
├── data/
│   └── X-ray input images
│
├── include/
│   ├── filters.h
│
├── output/
│   └── Generated outputs
│
├── src/
│   ├── filters.cu
│   ├── main.cu
│
├── Makefile
└── README.md
```

## Generated Output Files

The `output/` directory is intentionally kept small and presentation-ready:

| File | Purpose |
|---|---|
| `gaussian.png` | GPU Enhanced Gaussian blur result |
| `histogram.png` | GPU Enhanced histogram-equalized result |
| `sharpen.png` | GPU Enhanced sharpened result |
| `edge.png` | GPU Enhanced Sobel edge result |
| `preview.png` | Streamlit four-stage GPU preview |
| `preview_cpu.png` | Streamlit CPU preview |
| `comparison.csv` | CPU output comparison metrics |

CPU stage images and GPU1 stage images are temporary comparison artifacts. The Streamlit app recreates them when a new image is selected; they are not part of the final output set.

## Main Files

- `streamlit_app.py`: Streamlit UI, automatic CPU/GPU execution, timing, memory, and accuracy comparison.
- `timing_utils.py`: GPU timing parser and comparison summaries.
- `gpu_launcher.py`: Windows-to-WSL CUDA executable routing.
- `src/main.cu`: GPU Enhanced pipeline orchestration and CUDA timing.
- `src/main_gpu1.cu`: GPU1 baseline using straightforward CUDA kernels.
- `src/filters.cu`: optimized shared-memory and low-overhead Gaussian/filter kernels.
- `Makefile`: Linux CUDA build targets for `clearscan` and `gpu1`.

## Build on Linux with CUDA

On a Linux machine with CUDA and OpenCV development packages installed, run:

```bash
chmod +x build_linux_cuda.sh
./build_linux_cuda.sh
```

Or manually:

```bash
make clean
make
./clearscan data/BrokenBone1.jpg
```

### Requirements
- NVIDIA GPU with a supported CUDA toolkit
- `nvcc`
- OpenCV 4 development libraries (`libopencv-dev`)
- `pkg-config`

### Typical Ubuntu install

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev nvidia-cuda-toolkit
```

If you are using a container environment, the Apptainer workflow below is also supported:

```bash
apptainer shell --nv /scratch/csee147/csee147env.sif

make clean
make

./clearscan data/BrokenBone1.jpg
```

## Author

Isaiah Alcayde

University of California, Riverside

CS/EE 147 - GPU Programming

Spring 2026
>>>>>>> 9cf7ec9 (Initial project import)
