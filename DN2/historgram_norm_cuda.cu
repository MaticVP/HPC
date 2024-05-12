#include "./include/histogram_norm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "include/helper_cuda.h"

#include "include/stb_image.h"
#include "include/stb_image_write.h"
#include "include/cuda_functions.cuh"
#include "include/config.h"


__global__ void calc_new_pixel_value(int* cum_histogram, const int width, const int height)
{

    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (gidx == 0)
    {
        printf("DEVICE: START calculating new pixels\n");
    }

    int min = 0;
    int min_index = 0;
    int max_i = width * height;

    for (int color = 0; !min && color < 256; ++color) {
        min = cum_histogram[color + 256 * blockIdx.x];
        min_index = color;
    }

    for (int color = min_index+threadIdx.x; color < 256; color += blockDim.x) {
        //printf("Investigating block %d, thread %d\n", blockIdx.x, color);
        int value = cum_histogram[color + 256 * blockIdx.x];
        double upper_frac = value - min;
        int lower_frac = max_i - min;
        double frace = (upper_frac / lower_frac) * (255);
        int new_value = floor(frace);

        cum_histogram[color + 256 * blockIdx.x] = new_value;

    }
}

__global__ void set_new_values(unsigned char* image, int* cum_histogram, const int width, const int height, const int cpp)
{

    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (gidx == 0 & gidy == 0)
    {
        printf("DEVICE: START setting new values\n");
    }
    for (int y = gidx; y < height; y += blockDim.x * gridDim.x)
    {
        for (int x = gidy; x < width; x += blockDim.y * gridDim.y)
        {
            for (int c = 0; c < cpp; c += 1)
            {
                int channel_value = image[c + cpp * (x + y * width)];
                int new_color = cum_histogram[channel_value + 256 * c];
                image[c + cpp * (x + y * width)] = new_color;
            }
        }
    }
}



void histogram_norm_cuda(unsigned char* image, int width, int height, int cpp) {
    int* histograms_temp = (int*)malloc(cpp * 256 * sizeof(int));
    unsigned char* image_out = (unsigned char*)malloc(width * height * cpp * sizeof(unsigned char));
    memset(histograms_temp, 0, cpp * 256 * sizeof(int));

    int* min_index_per_channel;
    int* histograms;
    int* cumulative_histogram_pointer;
    unsigned char* image_cuda;

    // Setup Thread organization
    dim3 blockSize(16, 16);
    dim3 gridSize((height - 1) / blockSize.x + 1, (width - 1) / blockSize.y + 1);
    //dim3 gridSize(1, 1);

    dim3 blockSizeNewValue(32, 32);
    dim3 gridSizeNewValue((height - 1) / blockSizeNewValue.x + 1, (width - 1) / blockSizeNewValue.y + 1);

    //Cuda var
    int value = 0;
    // Allocate memory on the device
    checkCudaErrors(cudaMalloc(&image_cuda, width * height * cpp * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&histograms, cpp * 256 * sizeof(int)));
    checkCudaErrors(cudaMalloc(&cumulative_histogram_pointer, cpp * 256 * sizeof(int)));
    checkCudaErrors(cudaMalloc(&min_index_per_channel, 3 * sizeof(int)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy image to device and run kernel
    cudaEventRecord(start);
    checkCudaErrors(cudaMemcpy(image_cuda, image, width * height * cpp * sizeof(unsigned char), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(histograms, 0, cpp * 256 * sizeof(int)));

#if CUDA_USE_SHARED
    compute_histogram_shared << <gridSize, blockSize >> > (image_cuda, histograms, width, height, cpp);
#else
    compute_histogram_basic << <gridSize, blockSize >> > (image_cuda, histograms, width, height, cpp);
#endif
    //checkCudaErrors(cudaMemcpy(histograms_temp, histograms, cpp * 256 * sizeof(int), cudaMemcpyDeviceToHost));

#if CUDA_PARALLEL_CUM
    dim3 blockSizeCum(256);
    dim3 gridSizeCum(cpp);
    cumulative_histogram_parallel << <gridSizeCum, blockSizeCum >> > (cumulative_histogram_pointer, histograms,cpp);
#else
    dim3 blockSizeCum((1, 1));
    dim3 gridSizeCum(1);
    cumulative_histogram_basic << <gridSizeCum, blockSizeCum >> > (cumulative_histogram_pointer, histograms, cpp);
#endif

    checkCudaErrors(cudaMemcpy(histograms_temp, cumulative_histogram_pointer, cpp * 256 * sizeof(int), cudaMemcpyDeviceToHost));

    dim3 blockSize_new(256);
    dim3 numBlocks_new(cpp);

    calc_new_pixel_value << <numBlocks_new, blockSize_new >> > (cumulative_histogram_pointer, width, height);

    checkCudaErrors(cudaMemcpy(histograms_temp, cumulative_histogram_pointer, cpp * 256 * sizeof(int), cudaMemcpyDeviceToHost));

    set_new_values << <gridSize, blockSize >> > (image_cuda,cumulative_histogram_pointer, width, height, cpp);


    checkCudaErrors(cudaMemcpy(image, image_cuda, width * height * cpp * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    getLastCudaError("copy_image() execution failed\n");
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f milliseconds \n", milliseconds);


}
