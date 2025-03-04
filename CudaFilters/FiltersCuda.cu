#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <iostream>

__global__ void GrayScaleFilterCuda(unsigned char* input, unsigned char* output, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const int idx = (x * height + y) * 4;
        const auto r = input[idx];
        const auto g = input[idx + 1];
        const auto b = input[idx + 2];
        const auto gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        output[idx] = gray;
        output[idx + 1] = gray;
        output[idx + 2] = gray;
        output[idx + 3] = gray;
    }
}

extern "C" void ApplyGrayScaleFilterCuda(unsigned char* d_input, int width, int height)
{
    constexpr auto blockSizeDim = 16u;
    dim3 blockSize(blockSizeDim, blockSizeDim);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    unsigned char* d_output;
    unsigned char* deviceInput;
	const auto size = (width * height) * 4 * sizeof(unsigned char);
	cudaMalloc(reinterpret_cast<void**>(&deviceInput), size);
    cudaMemcpy(deviceInput, d_input, size, cudaMemcpyHostToDevice);
	cudaMalloc(reinterpret_cast<void**>(&d_output), size);
    GrayScaleFilterCuda << <gridSize, blockSize >> > (deviceInput, d_output, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(d_input, d_output, size, cudaMemcpyDeviceToHost);
	cudaFree(deviceInput);
	cudaFree(d_output);
}