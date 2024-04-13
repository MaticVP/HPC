#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0
#define CHANNELS 3

typedef struct
{
    int width;
    int height;
    int org_width;
    int numStrips;
    int stripHeight;
    int numUpTriangles;
    int numDownTriangles;
    int triangleWidth;
    int triangleHeight;
} StripsData;

// Ta je za uporabo pri izračun energije/odvodov slike.
// Stride pove kateri barvni kanal želiš. npr. v rgb prostoru bi z stride 0 dobil R kanal z stride 1 pa B kanal
unsigned char get_pixel(unsigned char *image, int y, int x,
                        int width, int height, int stride,
                        int org_width)
{

    if (x >= width)
        x = width - 1;

    if (0 > x)
        x = 0;

    if (y >= height)
        y = height - 1;

    if (0 > y)
        y = 0;

    return image[((x + org_width * y) * CHANNELS) + stride];
}

unsigned int get_pixel_cumulative_ver(unsigned int *image, int y, int x, int width, int height, int org_width)
{
    if (x >= width || 0 > x || y >= height || 0 > y)
        return UINT_MAX;

    return image[x + org_width * y];
}

void calc_image_energy(unsigned char *image_out, const unsigned char *image_in, unsigned int width,
                       unsigned int height, unsigned int org_width, unsigned int cpp)
{
#pragma omp parallel for
    for (unsigned int y = 0; y < height; y++)
    {
        for (unsigned int x = 0; x < width; x++)
        {
            unsigned int sum = 0;
            for (int c = 0; c < cpp; c++)
            {
                int Gx = -get_pixel(image_in, y - 1, x - 1, width, height, c, org_width) - 2 * get_pixel(image_in, y, x - 1, width, height, c, org_width) - get_pixel(image_in, y + 1, x - 1, width, height, c, org_width) + get_pixel(image_in, y - 1, x + 1, width, height, c, org_width) + 2 * get_pixel(image_in, y, x + 1, width, height, c, org_width) + get_pixel(image_in, y + 1, x + 1, width, height, c, org_width);

                int Gy = +get_pixel(image_in, y - 1, x - 1, width, height, c, org_width) + 2 * get_pixel(image_in, y - 1, x, width, height, c, org_width) + get_pixel(image_in, y - 1, x + 1, width, height, c, org_width) - get_pixel(image_in, y + 1, x - 1, width, height, c, org_width) - 2 * get_pixel(image_in, y + 1, x, width, height, c, org_width) - get_pixel(image_in, y + 1, x + 1, width, height, c, org_width);

                sum += (unsigned char)sqrt((Gx * Gx) + (Gy * Gy));
            }

            image_out[x + org_width * y] = sum / cpp;
        }
    }
}

unsigned int calculatePixelEnergy(int *out_image, int x, int y, int width, int height, int org_width)
{
    unsigned int bottom_left = get_pixel_cumulative_ver(out_image, y, x - 1, width, height, org_width);
    unsigned int bottom_mid = get_pixel_cumulative_ver(out_image, y, x, width, height, org_width);
    unsigned int bottom_right = get_pixel_cumulative_ver(out_image, y, x + 1, width, height, org_width);
    unsigned int smallest_value = bottom_mid;

    if (bottom_left < smallest_value)
    {
        smallest_value = bottom_left;
    }
    if (bottom_right < smallest_value)
    {
        smallest_value = bottom_right;
    }

    return smallest_value;
}

/**
 * We fill the bottom space with triangles from start to finish. Each triangle from the left side onwards will be
 * full sized (strip - 1 base width) except for the last one. Each thread calculates it's triangle start and end position
 * and for that triangle calculates  all the energies.
 */
void calculateBottomTriangleEnergy(
    StripsData stripData,
    int numTriangles,
    int currStrip,
    unsigned int *out_image,
    const unsigned char *energy_image)
{
    int height = stripData.height;
    int width = stripData.width;
    int org_width = stripData.org_width;
    int stripHeight = stripData.stripHeight;
    int triangleHeight = stripData.triangleHeight;
    int triangleWidth = stripData.triangleWidth;

    #pragma omp parallel for
    for (int i = 0; i < numTriangles; i++) {
        int from = i * triangleWidth;
        int to = from + triangleWidth;

        if(to > width) {
            to = width;
        }

        int currTriangleWidth = to - from;
        int currTriangleHeight = fmin(currTriangleWidth, triangleHeight);

        for (int k = from; k < to; k++) {
            unsigned int pixelEnergy = energy_image[org_width * (currStrip * stripHeight + stripHeight - 1) + k];

            if (currStrip < stripData.numStrips - 1) {
                unsigned int smallest_value = calculatePixelEnergy(out_image, k, currStrip * stripHeight + stripHeight, width, height, org_width);
                pixelEnergy = pixelEnergy + smallest_value;
            }
  
            out_image[org_width * (currStrip * stripHeight + stripHeight - 1) + k] = pixelEnergy;
        }

        for (int row = stripHeight - 2; row > 0; row--) {
            int reverseRow = (currTriangleHeight - row);

            int colTo = to - reverseRow;

            if(currTriangleWidth < triangleWidth) {
                colTo = to;
            }

            for (int col = from + reverseRow; col < colTo; col++) {
                unsigned int smallest_value = calculatePixelEnergy(out_image, col, currStrip * stripHeight + row + 1, width, height, org_width);
                unsigned int pixelEnergy = energy_image[org_width * (currStrip * stripHeight + row) + col] + smallest_value;

                out_image[org_width * (currStrip * stripHeight + row) + col] = pixelEnergy;
            }
        }
    }
}

/**
 * We fill the tp space with triangles from start to finish. First triangle will be half of the full triangle size, and the last one will also be partial.
 * All the triangles in the middle will be full sized. Each thread calculates it's triangle start and end position (different if it's first or last triangle) and
 * for that triangle calculates the energies
 */
void calculateTopTriangleEnergy(
    StripsData stripData,
    int numTriangles,
    int currStrip,
    unsigned int *out_image,
    const unsigned char *energy_image
)
{
    int height = stripData.height;
    int width = stripData.width;
    int org_width = stripData.org_width;
    int stripHeight = stripData.stripHeight;
    int triangleHeight = stripData.triangleHeight;
    int triangleWidth = stripData.triangleWidth;

    #pragma omp parallel for
    for (int i = 0; i < numTriangles; i++) {
        int from, to;

        if(i == 0) {
            from = 0;
            to = triangleWidth/2;
        } else {
            from = (i - 1) * triangleWidth + triangleWidth/2;
            to = from + triangleWidth;
        }

        if(to > width) {
            to = width;
        }

        int currTriangleWidth = to - from;
        int currTriangleHeight = fmin(currTriangleWidth, triangleHeight);
       
        for (int row = stripHeight - 2; row >= 0; row--) {
            int colFrom, colTo;

            if (from == 0) {
                colFrom = 0;
                colTo = to - row;
            } else if(to == width) {
                colFrom = from + row;
                colTo = width;
            } else {
                colFrom = from + row;
                colTo = to - row;
            }


            for (int col = colFrom; col < colTo; col++) {
                unsigned int smallest_value = calculatePixelEnergy(out_image, col, currStrip * stripHeight + row + 1, width, height, org_width);
                unsigned int pixelEnergy = energy_image[org_width * (currStrip * stripHeight + row) + col] + smallest_value;
  
                out_image[org_width * (currStrip * stripHeight + row) + col] = pixelEnergy;
            }
        }
    }
}

void calc_energy_cumulative_triangle(unsigned int *out_image, const unsigned char *energy_image,
                                     unsigned int width, unsigned int height, unsigned int org_width, int strip_height)
{
    StripsData stripsData;
    stripsData.width = width;
    stripsData.height = height;
    stripsData.stripHeight = strip_height;
    stripsData.org_width = org_width;
    stripsData.numStrips = (height + strip_height - 1) / strip_height;
    stripsData.triangleHeight = strip_height - 1;
    stripsData.triangleWidth = strip_height + stripsData.triangleHeight - 1;
    stripsData.numUpTriangles = (stripsData.width + stripsData.triangleHeight - 1) / stripsData.triangleWidth;
    stripsData.numDownTriangles = stripsData.numUpTriangles;

    // Number of Up/Down triangles is not the same and there is one more down triangle than up
    if (((stripsData.width % stripsData.triangleWidth) * stripsData.numUpTriangles) > (stripsData.triangleWidth / 2) || (stripsData.width % (stripsData.triangleWidth * stripsData.numUpTriangles)) == 0)
    {
        stripsData.numDownTriangles += 1;
    }
    
    for (int i = stripsData.numStrips - 1; i >= 0; i--)
    {
        calculateBottomTriangleEnergy(stripsData, stripsData.numUpTriangles, i, out_image, energy_image);
        #pragma omp barrier

        calculateTopTriangleEnergy(stripsData, stripsData.numDownTriangles, i, out_image, energy_image);
        #pragma omp barrier
    }
}

void seams_basic(unsigned int *path, unsigned int *energy_cumulative_image, unsigned char *image_in,
                 unsigned int width, unsigned int height, unsigned int org_width, int cpp)
{
    unsigned int smallest_index = UINT_MAX;
    unsigned int smallest_value = UINT_MAX;
    //find new smallest value
    //#pragma omp parallel for
    for (int i = 0; i < width; i++)
    {
        if (smallest_value > energy_cumulative_image[i])
        {
            smallest_value = energy_cumulative_image[i];
            smallest_index = i;
        }
    }

    //find path
    //#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        path[y] = smallest_index;

        unsigned int bottom_left = get_pixel_cumulative_ver(energy_cumulative_image, y + 1, smallest_index - 1, width, height, org_width);
        unsigned int bottom_mid = get_pixel_cumulative_ver(energy_cumulative_image, y + 1, smallest_index, width, height, org_width);
        unsigned int bottom_right = get_pixel_cumulative_ver(energy_cumulative_image, y + 1, smallest_index + 1, width, height, org_width);
        smallest_value = bottom_mid;

        if (bottom_left < smallest_value)
        {
            smallest_value = bottom_left;
            smallest_index = smallest_index - 1;
        }
        if (bottom_right < smallest_value)
        {
            smallest_index = smallest_index + 1;
        }
    }

    //#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        int seam_index = path[y];
        for (int x = seam_index; x < width - 1; x++)
        {
            for (int c = 0; c < cpp; c++)
            {
                image_in[(y * org_width + x) * cpp + c] = image_in[(y * org_width + x + 1) * cpp + c];
            }
        }
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

    // Set to number so that the division with height is 0
    // If 600x450, strip height should be 5, 10, 25, 75, etc.
    int strip_height = atoi(argv[4]); 

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
    unsigned char *image_out = (unsigned char *)malloc((width - num_seams) * height * cpp * sizeof(unsigned char));
    unsigned char *energy_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned int *energy_cumulative_image = (unsigned int *)malloc(width * height * sizeof(unsigned int));
    //unsigned int *topIndexOrder = (unsigned int *)malloc(width * sizeof(unsigned int));
    unsigned int *path = (unsigned int *)malloc(height * sizeof(unsigned int));

//Print the number of threads
#pragma omp parallel
    {
#pragma omp single
        printf("Using %d threads", omp_get_num_threads());
    }

    // Just copy the input image into output
    double start = omp_get_wtime();
    int org_width = width;

    for (int i = 0; i < num_seams; i++)
    {
        calc_image_energy(energy_image, image_in, width, height, org_width, cpp);
        //stbi_write_png("output_energyImage.png", width, height, 1, energy_image, org_width);
        calc_energy_cumulative_triangle(energy_cumulative_image, energy_image, width, height, org_width, strip_height);
        //stbi_write_png("output_energyCumImage.png", width, height, 1, energy_cumulative_image, org_width);
        seams_basic(path, energy_cumulative_image, image_in, width, height, org_width, cpp);
        width--;
    }

#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < cpp; c++)
            {
                image_out[(y * width + x) * cpp + c] = image_in[(y * org_width + x) * cpp + c];
            }
        }
    }
    //image_out = image_in;
    double stop = omp_get_wtime();
    printf(" -> time to copy: %f s\n", stop - start);
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
        stbi_write_png(image_out_name, width, height, cpp, image_out, (width)*cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, width, height, cpp, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, width, height, cpp, image_out);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", file_type);

    // Release the memory
    free(image_in);
    free(image_out);

    return 0;
}