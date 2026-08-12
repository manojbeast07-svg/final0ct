CUDA_BIN ?= $(shell dirname $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin))
export PATH := $(CUDA_BIN):$(PATH)

NVCC        = nvcc
CXX         ?= g++
NVCC_FLAGS  ?= -O3 -arch=sm_80

OPENCV_CFLAGS ?= $(shell pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS ?= $(shell pkg-config --libs opencv4 2>/dev/null)
ifeq ($(strip $(OPENCV_CFLAGS)$(OPENCV_LIBS)),)
OPENCV_CFLAGS = -I/usr/include/opencv4
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio
endif
OPENCV_FLAGS = $(OPENCV_CFLAGS) $(OPENCV_LIBS)
LD_FLAGS     = -lcudart -lstdc++

EXE = clearscan
GPU1_EXE = gpu1
OBJ = main.o filters.o

default: $(EXE)

gpu1: src/main_gpu1.cu
	$(NVCC) -O3 -arch=sm_80 -o $(GPU1_EXE) src/main_gpu1.cu -I./include -I./include/stb

main.o: src/main.cu include/filters.h
	$(NVCC) -c -o $@ src/main.cu $(NVCC_FLAGS) -I./include -I./include/stb $(OPENCV_FLAGS)

filters.o: src/filters.cu include/filters.h
	$(NVCC) -c -o $@ src/filters.cu $(NVCC_FLAGS) -I./include $(OPENCV_FLAGS)

$(EXE): $(OBJ)
	$(CXX) $(OBJ) -o $(EXE) $(OPENCV_FLAGS) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE) $(GPU1_EXE) output/*.png