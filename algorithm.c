#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <float.h>

#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MIN3(a, b, c) (MIN(a, MIN(b, c)))

void copy_image(unsigned char *image_og, const float *image_in, unsigned char *image_out, int width, int height, int cpp){
    for (int row = 0; row < height; row++) {
        int new_col = 0;
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            if (image_in[idx] != -1.0f) {
                int new_idx = row * (width - 1) + new_col;
                for (int c = 0; c < cpp; ++c) {
                    image_out[new_idx * cpp + c] = image_og[idx * cpp + c];
                }
                new_col++;
            }
        }
    }
}


void energy_map(const unsigned char *image_in, float *image_out, int width, int height){
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++)
        {
            int u = 1;
            int d = 1;
            int l = 1;
            int r = 1;

            // center
            int index = (i * width + j) * 3;

            if (i == 0){
                u = 0;
            }

            if (i == height - 1){
                d = 0;
            }

            if (j == 0){
                l = 0;
            }

            if (j == width - 1){
                r = 0;
            }

            int index_up = ((i - u) * width + j) * 3;
            int index_up_left = ((i - u) * width + (j - l)) * 3;
            int index_up_right = ((i - u) * width + (j + r)) * 3;
            int index_down = ((i + d) * width + j) * 3;
            int index_down_left = ((i + d) * width + (j - l)) * 3;
            int index_down_right = ((i + d) * width + (j + r)) * 3;
            int index_left = (i * width + (j - l)) * 3;
            int index_right = (i * width + (j + r)) * 3;

            float gx[3], gy[3];
            for (int c = 0; c < 3; c++) {
                gx[c] = -image_in[index_up_left + c] - 2 * image_in[index_left + c] - image_in[index_down_left + c]
                        + image_in[index_up_right + c] + 2 * image_in[index_right + c] + image_in[index_down_right + c];
                gy[c] = image_in[index_up_left + c] + 2 * image_in[index_up + c] + image_in[index_up_right + c]
                        - image_in[index_down_left + c] - 2 * image_in[index_down + c] - image_in[index_down_right + c];
            }

            float energyr = sqrtf(gx[0] * gx[0] + gy[0] * gy[0]);
            float energyg = sqrtf(gx[1] * gx[1] + gy[1] * gy[1]);
            float energyb = sqrtf(gx[2] * gx[2] + gy[2] * gy[2]);

            float energy = (energyr + energyg + energyb) / 3;


            image_out[index/3] = energy;
        }
    }
}


void find_paths(const float *image_in, float *image_out, int width, int height){
    for (int j = 0; j < width; j++) {
        int idx = (height - 1) * width + j;
        image_out[idx] = image_in[idx];
    }

    for (int i = height - 2; i >= 0; i--){
        for (int j = 0; j < width; j++){
            int l = 1;
            int r = 1;

            int index = (i * width + j);

            if (j == 0){
                l = 0;
            }

            if (j == width - 1){
                r = 0;
            }

            int index_down = (i + 1) * width + j;
            int index_down_left = (i + 1) * width + (j - l);
            int index_down_right = (i + 1) * width + (j + r);

            float distance = image_in[index] + MIN3(
                image_out[index_down_left],
                image_out[index_down],
                image_out[index_down_right]);

            image_out[index] = distance;
        }
    }
}


void remove_seam(unsigned char *image_og, float *image_in, unsigned char *image_out, int width, int height){
    float min_val = FLT_MAX;
    int min_index = 0;

    for (int i = 0; i < width; i++) {
        if (image_in[i] < min_val) {
            min_val = image_in[i];
            min_index = i;
        }
    }
    
    int index = min_index;

    for (int i = 1; i < height; i++)
    {
        int index_down = i * width + index;
        int index_down_left = i * width + (index - 1);
        int index_down_right = i * width + (index + 1);

        if (index != 0 && image_in[index_down_left] < image_in[index_down]){
            if (index == width - 1 || image_in[index_down_left] < image_in[index_down_right]){
                index -= 1;
            } else {
                index += 1;
            }
        } else if (index != width - 1 && image_in[index_down_right] < image_in[index_down]){
            index += 1;
        }

        image_in[i * width + index] = -1;
    }

    copy_image(image_og, image_in, image_out, width, height, 3);
}


int main(int argc, char *argv[]){
    if (argc < 3){
        printf("USAGE: alogorithm input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[255];
    char image_out_name[255];

    snprintf(image_in_name, 255, "%s", argv[1]);
    snprintf(image_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL){
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }

    int shortened = 1; // Number of removed seams
    printf("Loaded image %s of size %dx%d.\n", image_in_name, width, height);
    size_t datasize_float = width * height * sizeof(float);
    size_t datasize_carved = (width - shortened) * height * sizeof(unsigned char) * cpp;

    float *image_energy = malloc(datasize_float);
    float *image_paths = malloc(datasize_float);
    unsigned char *image_carved = malloc(datasize_carved);

    int seams = 128;

    for (int i = 1; i <= seams; i++) {
        // Calculate energy
        double start = omp_get_wtime();
        energy_map(image_in, image_energy, width, height);
        double stop = omp_get_wtime();
        //printf(" -> time to calculate energy: %f s\n", stop - start);

        // Calculate paths
        start = omp_get_wtime();
        find_paths(image_energy, image_paths, width, height);
        stop = omp_get_wtime();
        //printf(" -> time to paths: %f s\n", stop - start);

        // Remove seam
        start = omp_get_wtime();
        remove_seam(image_in, image_paths, image_carved, width, height);
        stop = omp_get_wtime();
        //printf(" -> time to remove: %f s\n", stop - start);

        width -= 1; // Decrease width after removing a seam

        image_in = image_carved;

        // Reallocate image_carved with the new size for the next seam
        datasize_carved = (width - 1) * height * sizeof(unsigned char) * cpp;
        image_carved = malloc(datasize_carved);
    }

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
        stbi_write_png(image_out_name, width, height, 3, image_in, width * 3); // * cpp, 1 = cpp
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, width, height, 3, image_carved, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, width, height, 3, image_carved);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", file_type);

    // Release the memory
    free(image_in);
    free(image_energy);
    free(image_paths);
    free(image_carved);

    return 0;
}
