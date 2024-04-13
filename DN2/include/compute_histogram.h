#ifndef DN2_COMPUTE_HISTOGRAM_H
#define DN2_COMPUTE_HISTOGRAM_H
__global__ void compute_histogram(int* histograms,unsigned char* image,int width,int height,int channels);
#endif //DN2_COMPUTE_HISTOGRAM_H
