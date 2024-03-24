#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0
#define CHANNELS 3

// Ta je za uporabo pri izračun energije/odvodov slike.
// Stride pove kateri barvni kanal želiš. npr. v rgb prostoru bi z stride 0 dobil R kanal z stride 1 pa B kanal
unsigned char get_pixel(unsigned char *image, int y,int x,
                        int width,int height,int stride,
                        int org_width,unsigned int cpp){

    if(x>=width)
        x=width-1;

    if(0>x)
        x=0;

    if(y>=height)
        y=height-1;

    if(0>y)
        y=0;

    return image[((x+org_width*y)*cpp)+stride];
}

unsigned int get_pixel_cumulative_ver(unsigned int *image, int y,int x,int width,int height,int org_width){

    if(x>=width || 0>x || y>=height || 0>y)
        return UINT_MAX;

    return image[x+org_width*y];
}

void gray_scale_image(unsigned char *image_out, const unsigned char *image_in,unsigned int width,
                      unsigned int height,unsigned int org_width,unsigned int cpp)
{
#pragma omp parallel for
    for (int i = 0; i < org_width * height; i++) {
        // Average the RGB channels to get grayscale value
        int index = i * cpp;
        unsigned char r = image_in[index];
        unsigned char g = image_in[index + 1];
        unsigned char b = image_in[index + 2];
        image_out[i] = (unsigned char)((r + g + b) / 3);
    }
}

void calc_image_energy(unsigned char *image_out, const unsigned char *image_in,unsigned int width,
                       unsigned int height,unsigned int org_width)
{
    #pragma omp parallel for collapse(2)
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            int Gx = -get_pixel(image_in,y-1,x-1,width,height,0,org_width,1)
                     -2*get_pixel(image_in,y,x-1,width,height,0,org_width,1)
                     -get_pixel(image_in,y+1,x-1,width,height,0,org_width,1)
                     +get_pixel(image_in,y-1,x+1,width,height,0,org_width,1)
                     +2*get_pixel(image_in,y,x+1,width,height,0,org_width,1)
                     +get_pixel(image_in,y+1,x+1,width,height,0,org_width,1);

            int Gy = +get_pixel(image_in,y-1,x-1,width,height,0,org_width,1)
                     +2*get_pixel(image_in,y-1,x,width,height,0,org_width,1)
                     +get_pixel(image_in,y-1,x+1,width,height,0,org_width,1)
                     -get_pixel(image_in,y+1,x-1,width,height,0,org_width,1)
                     -2*get_pixel(image_in,y+1,x,width,height,0,org_width,1)
                     -get_pixel(image_in,y+1,x+1,width,height,0,org_width,1);

            image_out[x+org_width*y] = (unsigned char )sqrt((Gx*Gx) + (Gy*Gy));
        }
    }

}

void calc_energy_cumulative_basic(unsigned int *out_image,const unsigned char* energy_image,
                                  unsigned int width,unsigned int height, unsigned int org_width){

#pragma omp parallel for
    for (int x = 0; x < width; x++) {
        out_image[x + org_width * (height-1)] = energy_image[x+org_width*(height-1)];
    }

    for (int y = height-2; y >= 0; y--) {
        //Threading je slow. Bug?
        // Če ne uporabim thread tukaj sem za 5s hitrejši
//#pragma omp parallel for
        for (int x = 0; x < width; x++) {
            unsigned int bottom_left  = get_pixel_cumulative_ver(out_image,  y + 1,x - 1, width, height,org_width);
            unsigned int bottom_mid   = get_pixel_cumulative_ver(out_image,   y + 1,  x, width, height,org_width);
            unsigned int bottom_right = get_pixel_cumulative_ver(out_image, y + 1, x + 1, width, height,org_width);
            unsigned int smallest_value = bottom_mid;
            if(bottom_left<smallest_value){
                smallest_value = bottom_left;
            }
            if(bottom_right<smallest_value){
                smallest_value = bottom_right;
            }
            unsigned int value = energy_image[x + org_width * y] + smallest_value;
            out_image[x + org_width * y] = value;
        }
//#pragma omp barrier
    }
}

/**
 * Finds the smallest values the we can use to start finding the path
 * */
void sortIndexs(unsigned int *indexList, unsigned int *valueList,
                unsigned int width, unsigned int num_seams){
    int i, j, min_idx;

    for (i = 0; i < num_seams; i++)
    {
        min_idx = i;
        for (j = i+1; j < width; j++)
            if (valueList[j] < valueList[min_idx])
                min_idx = j;

        if(min_idx != i) {
            indexList[i] = min_idx;
            unsigned int temp = valueList[i];
            valueList[i] = valueList[min_idx];
            valueList[min_idx] = temp;
        }
        else{
            indexList[i] = i;
        }
    }
}

void seams_greedy(unsigned int *path,unsigned int *indexList,unsigned char * pathOffset,
                  unsigned int* energy_cumulative_image,unsigned char* image_in, unsigned int width,
                  unsigned int height, unsigned int org_width, int cpp, unsigned int num_seams) {


    #pragma omp parallel for
    for (int i = 0; i < num_seams; i++) {
        path[i] = indexList[i];
    }

    //find path
    //pathOffset is used as a mask that prevents threads from picking the same path
    #pragma omp parallel for
    for(int i = 0; i < num_seams; i++) {
        unsigned int smallest_value = UINT_MAX;
        unsigned int smallest_index = path[i];
        for (int y = 0; y < height; y++) {
            path[(y) * num_seams + i] = smallest_index;
            pathOffset[smallest_index + org_width * (y)] = i+1;
            int org_index = path[y * num_seams + i];
            unsigned int bottom_left = get_pixel_cumulative_ver(energy_cumulative_image, y + 1, smallest_index - 1,
                                                                width, height, org_width);
            unsigned int bottom_mid = get_pixel_cumulative_ver(energy_cumulative_image, y + 1, smallest_index, width,
                                                               height, org_width);
            unsigned int bottom_right = get_pixel_cumulative_ver(energy_cumulative_image, y + 1, smallest_index + 1,
                                                                 width, height, org_width);
            smallest_value = bottom_mid;

            if ((bottom_left < smallest_value)
                && (org_index - 1) >= 0
                && (pathOffset[(org_index - 1) + org_width * (y + 1)] == 0)) {
                smallest_value = bottom_left;
                smallest_index = smallest_index - 1;
            }
            if ((bottom_right < smallest_value)
                && (org_index + 1) < width
                && ((pathOffset[(org_index + 1) + org_width * (y + 1)]) == 0)) {
                smallest_index = smallest_index + 1;
            }

        }
    }

    //With every shift you must correct the path since the pixels moved to the right
    for (int i = 0; i < num_seams; i++) {
        unsigned int twidth = width;
        #pragma omp parallel for
        for (int y = 0; y < height; y++) {
            int seam_index = path[(y) * num_seams + i]-i;
            for (int x = seam_index; x < width-1; x++) {
                for (int c = 0; c < cpp; c++) {
                    image_in[(y * org_width + x) * cpp + c] = image_in[(y * org_width + x + 1) * cpp + c];
                }
            }
        }
        width--;
    }
}




int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        printf("USAGE: sample input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[255];
    char image_out_name[255];

    snprintf(image_in_name, 255, "%s", argv[1]);
    snprintf(image_out_name, 255, "%s", argv[2]);
    int num_seams = atoi(argv[3]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL)
    {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", image_in_name, width, height);
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *image_out = (unsigned char *)malloc((width-num_seams) * height * cpp * sizeof(unsigned char));
    unsigned char *image_gray = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *energy_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned int *energy_cumulative_image = (unsigned int *)malloc(width * height * sizeof(unsigned int));
    unsigned int *indexList = (unsigned int *)malloc(width * sizeof(unsigned int));
    unsigned int *valueList = (unsigned int *)malloc(width * sizeof(unsigned int));
    unsigned char *pathVisited = (unsigned char *)malloc(height * width * sizeof(unsigned char ));
    unsigned int *path = (unsigned int *)malloc(num_seams * height* sizeof(unsigned int));
    unsigned int *smallestPathIndex = (unsigned int *)malloc(height* sizeof(unsigned int));

    memset(pathVisited, 0, height * width * sizeof(unsigned char ));
    memset(smallestPathIndex,UINT_MAX,height * sizeof(unsigned int));
    //Print the number of threads
#pragma omp parallel
    {
#pragma omp single
        printf("Using %d threads",omp_get_num_threads());
    }


    // Just copy the input image into output
    double start = omp_get_wtime();
    int org_width = width;

    gray_scale_image(image_gray, image_in, width, height,org_width,cpp);
    //stbi_write_png("output_grayImage.png", width, height, 1, image_gray, org_width);
    calc_image_energy(energy_image, image_gray, width, height,org_width);
    stbi_write_png("output_energyImage.png", width, height, 1, energy_image, org_width);
    calc_energy_cumulative_basic(energy_cumulative_image, energy_image, width, height,org_width);
    //stbi_write_png("output_energyCumImage.png", width, height, 1, energy_cumulative_image, org_width);

#pragma omp parallel for
    for (int x = 0; x < width; x++) {
        valueList[x] = energy_cumulative_image[x];
    }
    // sort values to use for index paths
    sortIndexs(indexList, valueList, width, num_seams);
    seams_greedy(path,indexList, pathVisited,
                 energy_cumulative_image,image_in, width,height, org_width,
                 cpp, num_seams);

    width-=num_seams;



#pragma omp parallel for collapse(3)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < cpp; c++) {
                image_out[(y * width + x) * cpp + c] = image_in[(y * org_width + x) * cpp + c];
            }
        }
    }
    //image_out = image_in;
    double stop = omp_get_wtime();
    printf(" -> time to copy: %f s\n",stop-start);
    // Write the output image to file
    char image_out_name_temp[255];
    strncpy(image_out_name_temp, image_out_name, 255);
    char *token = strtok(image_out_name_temp, ".");
    char *file_type = NULL;
    while (token != NULL)
    {
        file_type = token;
        token = strtok(NULL, ".");
    }

    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, width, height, cpp, image_out, (width) * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, width, height, cpp, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, width, height, cpp, image_out);
        //stbi_write_bmp(image_out_name, org_width, height, cpp, image_in);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", file_type);

    // Release the memory
    free(image_in);
    free(image_out);


    return 0;
}
