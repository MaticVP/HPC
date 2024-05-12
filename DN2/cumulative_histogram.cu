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
    __shared__ int temp[256];

    const int n = 256;

    int thid = threadIdx.x;
    int offset = 1;


    int ai = 0;
    int bi = 0;
    int offsetA = 0;
    int offsetB = 0;

    for (int i = thid; i < n; i += blockDim.x) {

        int ai = thid;
        int bi = thid + (n / 2);

        offsetA = CONFLICT_FREE_OFFSET(ai);
        offsetB = CONFLICT_FREE_OFFSET(ai);

        temp[ai + offsetA] = histograms[ai + 256 * blockIdx.x];
        temp[bi + offsetB] = histograms[(bi + 1) + 256 * blockIdx.x];
    }

    //up-sweep
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[ai] = temp[bi];
        }
        offset *= 2;
    }


    if (thid == 0) { 
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    //down-sweep
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    cum_histogram[(ai * thid) + 256 * blockIdx.x] = temp[ai + offsetA];
    cum_histogram[(bi * thid + 1) + 256 * blockIdx.x] = temp[bi + offsetB];
     /*
    cum_histogram[(thid) + 256 * blockIdx.x] = temp[(thid)];
    cum_histogram[(thid + 1) + 256 * blockIdx.x] = temp[(thid + 1)];
    */
}