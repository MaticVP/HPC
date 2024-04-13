//
// Created by matic on 8. 04. 2024.
//

/*
 * In this file add function definistions for all functions that are related to CUDA
 */
#ifndef DN2_CUM_HIST_CUDA_H
#define DN2_CUM_HIST_CUDA_H
__global__ void cumulative_histogram_basic(int* cum_histogram, int* histograms, const int cpp);
__global__ void cumulative_histogram_parallel(int* cum_histogram, int* histograms, const int cpp);
__global__ void compute_histogram_basic(const unsigned char* image, int* histograms, const int width, const int height, const int cpp);
__global__ void compute_histogram_shared(const unsigned char* image, int* histograms, const int width, const int height, const int cpp);
__global__ void calc_new_pixel_value(int* cum_histogram, const int width, const int height);
__global__ void set_new_values(unsigned char* image, int* cum_histogram, const int width, const int height, const int cpp);
#endif //DN2_CONFIG_H
