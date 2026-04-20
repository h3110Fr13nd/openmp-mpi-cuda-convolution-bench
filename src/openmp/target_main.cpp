#include "convolution.hpp"

#include <chrono>
#include <cmath>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
Image convolve_omp_target(const Image &input,
                          const std::vector<float> &kernel,
                          int ksize) {
    Image output;
    output.width = input.width;
    output.height = input.height;
    output.data.assign(input.width * input.height, 0.0f);

#ifdef _OPENMP
    int devices = omp_get_num_devices();
    if (devices <= 0) {
        return convolve_sequential(input, kernel, ksize);
    }

    const int width = input.width;
    const int height = input.height;
    const int radius = ksize / 2;
    const float *in_ptr = input.data.data();
    const float *k_ptr = kernel.data();
    float *out_ptr = output.data.data();

    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: in_ptr[0:width*height], k_ptr[0:ksize*ksize]) \
        map(from: out_ptr[0:width*height])
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    int ix = x + kx - radius;
                    int iy = y + ky - radius;
                    float pixel = 0.0f;
                    if (ix >= 0 && iy >= 0 && ix < width && iy < height) {
                        pixel = in_ptr[iy * width + ix];
                    }
                    sum += pixel * k_ptr[ky * ksize + kx];
                }
            }
            out_ptr[y * width + x] = sum;
        }
    }
#else
    output = convolve_sequential(input, kernel, ksize);
#endif

    return output;
}

Image sobel_omp_target(const Image &input) {
    auto kx = make_sobel_x_kernel();
    auto ky = make_sobel_y_kernel();

#ifdef _OPENMP
    int devices = omp_get_num_devices();
    if (devices <= 0) {
        return sobel_magnitude_sequential(input);
    }
#endif

    Image gx = convolve_omp_target(input, kx, 3);
    Image gy = convolve_omp_target(input, ky, 3);

    Image output;
    output.width = input.width;
    output.height = input.height;
    output.data.resize(input.width * input.height);

#ifdef _OPENMP
    const int total = input.width * input.height;
    const float *gx_ptr = gx.data.data();
    const float *gy_ptr = gy.data.data();
    float *out_ptr = output.data.data();

    #pragma omp target teams distribute parallel for map(to: gx_ptr[0:total], gy_ptr[0:total]) map(from: out_ptr[0:total])
    for (int i = 0; i < total; ++i) {
        float mag = sqrtf(gx_ptr[i] * gx_ptr[i] + gy_ptr[i] * gy_ptr[i]);
        out_ptr[i] = mag;
    }
#else
    for (int i = 0; i < input.width * input.height; ++i) {
        float mag = std::sqrt(gx.data[i] * gx.data[i] + gy.data[i] * gy.data[i]);
        output.data[i] = mag;
    }
#endif

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
                output = sobel_omp_target(input);
            } else {
                auto kernel = make_blur_kernel(options.kernel_size);
                output = convolve_omp_target(input, kernel, options.kernel_size);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (!options.no_output) {
            if (options.output_path.empty()) {
                options.output_path = "results/images/omp_target_output.pgm";
            }
            save_pgm(output, options.output_path);
        }

#ifdef _OPENMP
        std::cout << "openmp_target_devices," << omp_get_num_devices() << "\n";
#endif
        std::cout << "openmp_target_elapsed_seconds," << elapsed.count() << "\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
