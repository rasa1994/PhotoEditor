#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <iostream>

__global__ void grayscale_filter(unsigned char* input, unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        output[y * width + x] = gray;
    }
}

extern "C" void apply_grayscale_filter(unsigned char* d_input, int width, int height)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    unsigned char* d_output = nullptr;
	cudaMalloc(reinterpret_cast<void**>(&d_output), (width * height) * 4 * sizeof(unsigned char));
    grayscale_filter << <gridSize, blockSize >> > (d_input, d_output, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(d_input, d_output, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}