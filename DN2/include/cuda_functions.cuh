/*
 * In this file add function definistions for all functions that are related to CUDA
 */
#ifndef DN2_CUM_HIST_CUDA_H
#define DN2_CUM_HIST_CUDA_H
#include "config.h"
__global__ void cumulative_histogram_basic(int* cum_histogram, int* histograms, const int cpp);
__global__ void cumulative_histogram_parallel(int* cum_histogram, int* histograms, const int cpp);
__global__ void compute_histogram_basic(const unsigned char* image, int* histograms, const int width, const int height, const int cpp);
__global__ void compute_histogram_shared(const unsigned char* image, int* histograms, const int width, const int height, const int cpp);
__global__ void calc_new_pixel_value(int* cum_histogram, const int width, const int height);
__global__ void set_new_values(unsigned char* image, int* cum_histogram, const int width, const int height, const int cpp);

#define NUM_BANKS 32
#define LOG_NUM_BANKS 4
#if ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif
#endif //DN2_CONFIG_H
