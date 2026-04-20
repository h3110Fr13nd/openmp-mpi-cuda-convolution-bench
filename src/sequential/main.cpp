#include "convolution.hpp"

#include <chrono>
#include <iostream>

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
                output = sobel_magnitude_sequential(input);
            } else {
                auto kernel = make_blur_kernel(options.kernel_size);
                output = convolve_sequential(input, kernel, options.kernel_size);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (!options.no_output) {
            if (options.output_path.empty()) {
                options.output_path = "results/images/seq_output.pgm";
            }
            save_pgm(output, options.output_path);
        }

        std::cout << "sequential_elapsed_seconds," << elapsed.count() << "\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
