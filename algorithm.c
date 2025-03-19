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
#define MAX(a, b) ((a) > (b) ? (a) : (b))

void copy_image(unsigned char *image_og, const float *image_in, unsigned char *image_out, int width, int height, int cpp){
    
    #ifdef ENABLE_OMP
        #pragma omp parallel for
    #endif
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

void copy_image_greedy(unsigned char *image_og, const float *image_in, unsigned char *image_out, int width, int height, int cpp, int num_seams) {
   
    int new_width = width - num_seams;
    
    #ifdef ENABLE_OMP
        #pragma omp parallel for
    #endif
    for (int row = 0; row < height; row++) {
        int new_col = 0;
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            if (image_in[idx] != -1.0f) {
                int new_idx = row * new_width + new_col;
                for (int c = 0; c < cpp; ++c) {
                    image_out[new_idx * cpp + c] = image_og[idx * cpp + c];
                }
                new_col++;
            }
        }
    }
}

void energy_map(const unsigned char *image_in, float *image_out, int width, int height){
    
    #ifdef ENABLE_OMP
        #pragma omp parallel for collapse(2)
    #endif
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

void calculate_pixel_energy(const unsigned char *image_in, float *image_out, int width, int height, int i, int j) {
    int u = 1;
    int d = 1;
    int l = 1;
    int r = 1;

    // center
    int index = (i * width + j) * 3;

    if (i == 0) {
        u = 0;
    }

    if (i == height - 1) {
        d = 0;
    }

    if (j == 0) {
        l = 0;
    }

    if (j == width - 1) {
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

    image_out[i * width + j] = energy;
}

void energy_map_triangle(const unsigned char *image_in, float *image_out, int width, int height, int strip_height) {

    // Number of strips needed to cover the image
    const int num_strips = (height + strip_height - 1) / strip_height;
    const int triangle_height = strip_height - 1;
    const int triangle_width = 2 * triangle_height;

    // Need to process each strip sequentially, bottom up
    for (int strip = 0; strip < num_strips; strip++) {
        int strip_start = strip * strip_height;
        int strip_end = MIN(strip_start + strip_height, height);
        int strip_actual_height = strip_end - strip_start;

        #ifdef ENABLE_OMP
            #pragma omp parallel for schedule(dynamic)
        #endif
        for (int triangle = 0; triangle < width; triangle += triangle_width) {
            int triangle_end = MIN(triangle + triangle_width, width - 1);

            for (int i = strip_start; i < strip_end; i++) {
                int row_offset = i - strip_start;
                int start_col = triangle + row_offset;
                int end_col = MIN(triangle + triangle_width - row_offset, triangle_end);

                if (start_col >= width) continue;

                for (int j = start_col; j <= end_col; j++) {
                    calculate_pixel_energy(image_in, image_out, width, height, i, j);
                }
            }
        }

        // Synchronization
        #ifdef ENABLE_OMP
            #pragma omp barrier
        #endif

        #ifdef ENABLE_OMP
            #pragma omp parallel for schedule(dynamic)
        #endif
        for (int triangle = triangle_width; triangle < width; triangle += triangle_width) {
            int triangle_start = triangle - triangle_width;
            int first_downward_row = strip_start + 1;

            for (int row_offset = triangle_height - 1; row_offset >= 0; row_offset--) {
                int i = strip_end - 1 - row_offset;

                if (i < first_downward_row) continue;

                int start_col = triangle_start;
                int end_col = MIN(triangle + triangle_width - row_offset, width - 1);

                if (start_col > end_col) continue;

                for (int j = start_col; j <= end_col; j++) {
                    calculate_pixel_energy(image_in, image_out, width, height, i, j);
                }
            }
        }

        // Synchronization
        #ifdef ENABLE_OMP
            #pragma omp barrier
        #endif
    }
}


void remove_seam_greedy(unsigned char *image_og, float *image_in, unsigned char *image_out, int width, int height, int num_seams) {
    
    num_seams = MIN(num_seams, width - 1);
    
    // We compute number of column strips based on batch seams to be removed
    int num_strips = num_seams;
    int strip_width = width / num_strips;
    
    // Seam paths
    int *seam_paths = (int *)malloc(height * num_seams * sizeof(int));
    if (!seam_paths) {
        fprintf(stderr, "Memory allocation failed for seam paths!\n");
        exit(EXIT_FAILURE);
    }
    
    // Find minimum energy in each of the strips
    #pragma omp parallel for
    for (int s = 0; s < num_seams; s++) {
        int strip_start = s * strip_width;
        int strip_end = (s == num_strips - 1) ? width : strip_start + strip_width;
        
        
        float min_val = FLT_MAX;
        int min_index = strip_start;
        
        for (int j = strip_start; j < strip_end; j++) {
            if (image_in[j] < min_val) {
                min_val = image_in[j];
                min_index = j;
            }
        }
        
        seam_paths[s * height] = min_index;
    }
    
    // Go down each seam in parallel
    #pragma omp parallel for
    for (int s = 0; s < num_seams; s++) {
        for (int i = 1; i < height; i++) {
            int prev_index = seam_paths[s * height + (i - 1)];
            float min_energy = FLT_MAX;
            int best_index = prev_index;
            
            for (int offset = -1; offset <= 1; offset++) {
                int curr_index = prev_index + offset;
                
                // Skip if out of bounds
                if (curr_index < 0 || curr_index >= width) continue;

                int pos = i * width + curr_index;
                
                if (image_in[pos] < min_energy) {
                    min_energy = image_in[pos];
                    best_index = curr_index;
                }
            }
            
            seam_paths[s * height + i] = best_index;
        }
    }
    
    // Marking seams for removal
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < height; i++) {
        for (int s = 0; s < num_seams; s++) {
            int seam_col = seam_paths[s * height + i];
            int index = i * width + seam_col;
            image_in[index] = FLT_MAX;
        }
    }

    copy_image_greedy(image_og, image_in, image_out, width, height, 3, num_seams);
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

    // Default values
    int seams = 128;
    int seam_batch_size = 1;
    int strip_size = -1;

    // Parse optional arguments
    for (int i = 3; i < argc; i++) {
        if (strncmp(argv[i], "seam_batch_size=", 16) == 0) {
            seam_batch_size = atoi(argv[i] + 16);
        } else if (strncmp(argv[i], "strip_size=", 11) == 0) {
            strip_size = atoi(argv[i] + 11);
        }
    }

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

    double total_energy_time = 0.0;
    double total_paths_time = 0.0;
    double total_remove_time = 0.0;
    double total_iteration_time = 0.0;

    double full_time_start = omp_get_wtime();

    for (int i = 1; i <= seams; i += seam_batch_size) {

        double iteration_start = omp_get_wtime();

        // Calculate energy
        double start = omp_get_wtime();
        // Energy map with triangles
        if (strip_size > 0) {
            energy_map_triangle(image_in, image_energy, width, height, strip_size);
        } else {
            // Normal energy map
            energy_map(image_in, image_energy, width, height);
        }
        double stop = omp_get_wtime();
        total_energy_time += stop - start;
        //printf(" -> time to calculate energy: %f s\n", stop - start);

        // Calculate paths
        start = omp_get_wtime();
        find_paths(image_energy, image_paths, width, height);
        stop = omp_get_wtime();
        total_paths_time += stop - start;
        //printf(" -> time to paths: %f s\n", stop - start);

        int seams_to_remove = (i + seam_batch_size <= seams) ? seam_batch_size : (seams - i + 1);

        // Remove seam
        start = omp_get_wtime();

        // Greedy seam removal
        if (seam_batch_size > 1) {
            remove_seam_greedy(image_in, image_paths, image_carved, width, height, seam_batch_size);
        } else {
            // Normal seam removal
            remove_seam(image_in, image_paths, image_carved, width, height);
        }
        stop = omp_get_wtime();
        total_remove_time += stop - start;
        //printf(" -> time to remove: %f s\n", stop - start);
        double iteration_end = omp_get_wtime();
        total_iteration_time += iteration_end - iteration_start;

        width -= seams_to_remove; // Decrease width after removing a seam

        unsigned char *temp = image_in;
        image_in = image_carved;
        free(temp);

        if (width > 0) {
            datasize_carved = width * height * sizeof(unsigned char) * cpp;
            image_carved = (unsigned char *)malloc(datasize_carved);
            if (!image_carved) {
                fprintf(stderr, "Memory allocation failed!\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    double full_time_end = omp_get_wtime();

    // Print average times
    printf("\n==== Benchmark Results ====\n");
    printf("Total time taken: %f s\n", full_time_end - full_time_start);
    printf("Average Energy Calculation Time per Seam: %f s\n", total_energy_time / seams);
    printf("Average Path Calculation Time per Seam: %f s\n", total_paths_time / seams);
    printf("Average Seam Removal Time per Seam: %f s\n", total_remove_time / seams);
    printf("Average Total Iteration Time per Seam: %f s\n", total_iteration_time / seams);

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
