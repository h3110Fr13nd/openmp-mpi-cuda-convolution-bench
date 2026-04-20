#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Image {
    int width = 0;
    int height = 0;
    std::vector<float> data;

    float &at(int x, int y) { return data[y * width + x]; }
    float at(int x, int y) const { return data[y * width + x]; }
};

struct Options {
    std::string input_path;
    std::string output_path;
    std::string filter = "blur";
    int kernel_size = 3;
    int width = 1024;
    int height = 1024;
    int iterations = 1;
    bool no_output = false;
};

Image load_pgm(const std::string &path);
void save_pgm(const Image &image, const std::string &path);
Image generate_image(int width, int height);

std::vector<float> make_blur_kernel(int size);
std::vector<float> make_sobel_x_kernel();
std::vector<float> make_sobel_y_kernel();

Options parse_args(int argc, char **argv);

float get_pixel(const Image &image, int x, int y);

void convolve_range(const Image &input,
                    const std::vector<float> &kernel,
                    int ksize,
                    int y_start,
                    int y_end,
                    Image &output);

Image convolve_sequential(const Image &input,
                          const std::vector<float> &kernel,
                          int ksize);

Image sobel_magnitude_sequential(const Image &input);
