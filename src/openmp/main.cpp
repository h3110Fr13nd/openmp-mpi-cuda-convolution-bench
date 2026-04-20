#include "convolution.hpp"

#include <chrono>
#include <cmath>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
Image convolve_openmp(const Image &input,
                      const std::vector<float> &kernel,
                      int ksize) {
    Image output;
    output.width = input.width;
    output.height = input.height;
    output.data.assign(input.width * input.height, 0.0f);

    int radius = ksize / 2;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    int ix = x + kx - radius;
                    int iy = y + ky - radius;
                    float pixel = get_pixel(input, ix, iy);
                    float kval = kernel[ky * ksize + kx];
                    sum += pixel * kval;
                }
            }
            output.at(x, y) = sum;
        }
    }

    return output;
}

Image sobel_openmp(const Image &input) {
    auto kx = make_sobel_x_kernel();
    auto ky = make_sobel_y_kernel();

    Image gx = convolve_openmp(input, kx, 3);
    Image gy = convolve_openmp(input, ky, 3);

    Image output;
    output.width = input.width;
    output.height = input.height;
    output.data.resize(input.width * input.height);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < input.width * input.height; ++i) {
        float mag = std::sqrt(gx.data[i] * gx.data[i] + gy.data[i] * gy.data[i]);
        output.data[i] = mag;
    }

    return output;
}
} // namespace

int main(int argc, char **argv) {
    try {
        Options options = parse_args(argc, argv);

        Image input = options.input_path.empty()
                          ? generate_image(options.width, options.height)
                          : load_pgm(options.input_path);

        Image output;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < options.iterations; ++i) {
            if (options.filter == "sobel") {
                output = sobel_openmp(input);
            } else {
                auto kernel = make_blur_kernel(options.kernel_size);
                output = convolve_openmp(input, kernel, options.kernel_size);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (!options.no_output) {
            if (options.output_path.empty()) {
                options.output_path = "results/images/omp_output.pgm";
            }
            save_pgm(output, options.output_path);
        }

#ifdef _OPENMP
        std::cout << "openmp_threads," << omp_get_max_threads() << "\n";
#endif
        std::cout << "openmp_elapsed_seconds," << elapsed.count() << "\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
