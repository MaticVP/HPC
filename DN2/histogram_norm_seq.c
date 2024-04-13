//
// Created by matic on 8. 04. 2024.
//
#include "./include/histogram_norm.h"
#include "include/config.h"
#include "stdlib.h"
#include <math.h>
#include <time.h>
#include <omp.h>

void histogram_norm(unsigned char* image,int width, int height, int cpp){
    int* histograms = (int* )malloc(cpp * 256 * sizeof(int));
    int* cumulative_histogram = (int* )malloc(cpp * 256 * sizeof(int));
    memset(histograms,0,cpp * 256 * sizeof(int));

    double start_time, end_time;

    start_time = omp_get_wtime();

    //Normal histogram
    for(int y=0;y<height;y++){
        for(int x=0;x<width;x++){
            for (int c = 0; c < cpp; ++c) {
                int channel_value = image[c + cpp * (x + y * width)];
                histograms[channel_value+256*c]++;
            }
        }
    }


    //Calculate Cumulative
    for (int c = 0; c < cpp; ++c) {
        int value = 0;
        for (int color = 0; color < 256; ++color) {
            value+=histograms[color+256*c];
            cumulative_histogram[color+256*c]=value;
        }
    }

    //Calc new value
    for (int c = 0; c < cpp; ++c) {
        int min = 0;
        int max = cumulative_histogram[255+256*c];
        int i = 0;

        for (int color = 0; min==0 && color<256; ++color) {
            min = cumulative_histogram[color+256*c];
            i=color;
        }

        for (int color = i; color < 256; ++color) {
            double upper_frac = cumulative_histogram[color+256*c]-min;
            int lower_frac = max-min;
            double frace = (upper_frac/lower_frac)*(255);
            int new_value = floor(frace);
            cumulative_histogram[color+256*c] = new_value;
        }
    }

    //place new values
    for(int y=0;y<height;y++){
        for(int x=0;x<width;x++){
            for (int c = 0; c < cpp; ++c) {
                int channel_value = image[c + cpp * (x + y * width)];
                int new_color = cumulative_histogram[channel_value+256*c];
                image[c + cpp * (x + y * width)] = new_color;
            }
        }
    }

    end_time = omp_get_wtime();
    double elapsed_time = (end_time - start_time)*1000.0;


    printf("Execution time: %.3f milliseconds\n", elapsed_time);

}
