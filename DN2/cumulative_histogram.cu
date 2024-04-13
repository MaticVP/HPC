#include <cuda_runtime.h>
#include <cuda.h>

#include "include/cuda_functions.cuh"

/*
 * In this file add function that are related to cumulative histogram
 */
__global__ void cumulative_histogram_basic(int* cum_histogram, int* histograms, const int cpp)
{

    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (gidx == 0 & gidy == 0)
    {
        for (int c = 0; c < cpp; ++c) {
            int value = 0;
            int found_first_no_zero = 0;
            for (int color = 0; color < 256; ++color) {
                value += histograms[color + 256 * c];
                cum_histogram[color + 256 * c] = value;
            }
        }
    }
}

__global__ void cumulative_histogram_parallel(int* cum_histogram, int* histograms, const int cpp)
{



}