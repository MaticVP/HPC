#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#include "include/cuda_functions.cuh"

/*
 * In this file add function that are related to histogram compute
 */
__global__ void compute_histogram_basic(const unsigned char* image, int* histograms, const int width, const int height, const int cpp)
{

    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (gidx == 0 & gidy == 0)
    {
        printf("DEVICE: START Histograms\n");
    }

    for (int y = gidx; y < height; y += blockDim.x * gridDim.x)
    {
        for (int x = gidy; x < width; x += blockDim.y * gridDim.y)
        {
            for (int c = 0; c < cpp; c += 1)
            {
                int channel_value = image[c + cpp * (x + y * width)];
                atomicAdd(&histograms[channel_value + 256 * c], 1);
            }
        }
    }

}

__global__ void compute_histogram_shared(const unsigned char* image, int* histograms, const int width, const int height, const int cpp)
{



}