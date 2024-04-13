#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./include/stb_image.h"
#include "./include/stb_image_write.h"
#include "./include/config.h"
#include "./include/histogram_norm_cuda.h"
extern "C" {
#include "./include/histogram_norm.h"
}

extern void histogram_norm_cuda(unsigned char* im, int width, int height, int cpp);
extern void histogram_norm(unsigned char* im, int width, int height, int cpp);

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

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL)
    {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", image_in_name, width, height);
    //const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *image_out = (unsigned char *)malloc(width * height * cpp * sizeof(unsigned char));

#if !CUDA
    histogram_norm(image_in,width,height,cpp);
#else
    histogram_norm_cuda(image_in, width, height, cpp);
#endif

    //image_out = image_in;
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
        stbi_write_png(image_out_name, width, height, cpp, image_in, width * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, width, height, cpp, image_in, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, width, height, cpp, image_in);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", file_type);

    // Release the memory
    free(image_in);
    //free(image_out);

    return 0;
}
