#include "convolution.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {
std::string next_token(std::istream &in) {
    std::string token;
    while (in >> token) {
        if (!token.empty() && token[0] == '#') {
            std::string line;
            std::getline(in, line);
            continue;
        }
        return token;
    }
    return "";
}

void validate_kernel_size(int size) {
    if (size < 1 || size % 2 == 0) {
        throw std::runtime_error("Kernel size must be odd and >= 1.");
    }
}

int to_int(const std::string &value, const std::string &name) {
    try {
        return std::stoi(value);
    } catch (...) {
        throw std::runtime_error("Invalid integer for " + name + ": " + value);
    }
}
} // namespace

Image load_pgm(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open input image: " + path);
    }

    std::string magic = next_token(in);
    if (magic != "P5") {
        throw std::runtime_error("Only binary PGM (P5) supported: " + path);
    }

    int width = to_int(next_token(in), "width");
    int height = to_int(next_token(in), "height");
    int maxval = to_int(next_token(in), "maxval");
    if (maxval <= 0 || maxval > 255) {
        throw std::runtime_error("Unsupported maxval in PGM: " + std::to_string(maxval));
    }

    in.get();

    Image image;
    image.width = width;
    image.height = height;
    image.data.resize(width * height);

    std::vector<unsigned char> buffer(width * height);
    in.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
    if (!in) {
        throw std::runtime_error("Failed to read pixel data from: " + path);
    }

    for (size_t i = 0; i < buffer.size(); ++i) {
        image.data[i] = static_cast<float>(buffer[i]);
    }

    return image;
}

void save_pgm(const Image &image, const std::string &path) {
    std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open output image: " + path);
    }

    out << "P5\n" << image.width << " " << image.height << "\n255\n";

    std::vector<unsigned char> buffer(image.width * image.height);
    for (size_t i = 0; i < buffer.size(); ++i) {
        float value = std::clamp(image.data[i], 0.0f, 255.0f);
        buffer[i] = static_cast<unsigned char>(value);
    }

    out.write(reinterpret_cast<const char *>(buffer.data()), buffer.size());
}

Image generate_image(int width, int height) {
    Image image;
    image.width = width;
    image.height = height;
    image.data.resize(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float gx = static_cast<float>(x) / static_cast<float>(width - 1);
            float gy = static_cast<float>(y) / static_cast<float>(height - 1);
            float value = 255.0f * (0.6f * gx + 0.4f * gy);
            image.at(x, y) = value;
        }
    }
    return image;
}

std::vector<float> make_blur_kernel(int size) {
    validate_kernel_size(size);
    std::vector<float> kernel(size * size, 1.0f);
    float scale = 1.0f / static_cast<float>(size * size);
    for (auto &v : kernel) {
        v *= scale;
    }
    return kernel;
}

std::vector<float> make_sobel_x_kernel() {
    return {
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f,
    };
}

std::vector<float> make_sobel_y_kernel() {
    return {
        -1.0f, -2.0f, -1.0f,
         0.0f,  0.0f,  0.0f,
         1.0f,  2.0f,  1.0f,
    };
}

Options parse_args(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            options.input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            options.output_path = argv[++i];
        } else if (arg == "--filter" && i + 1 < argc) {
            options.filter = argv[++i];
        } else if (arg == "--kernel-size" && i + 1 < argc) {
            options.kernel_size = to_int(argv[++i], "kernel-size");
        } else if (arg == "--width" && i + 1 < argc) {
            options.width = to_int(argv[++i], "width");
        } else if (arg == "--height" && i + 1 < argc) {
            options.height = to_int(argv[++i], "height");
        } else if (arg == "--iterations" && i + 1 < argc) {
            options.iterations = to_int(argv[++i], "iterations");
        } else if (arg == "--no-output") {
            options.no_output = true;
        } else if (arg == "--help") {
            std::cout << "Usage: [--input path] [--output path] [--filter blur|sobel] "
                         "[--kernel-size odd] [--width N] [--height N] [--iterations N] [--no-output]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.kernel_size % 2 == 0) {
        throw std::runtime_error("Kernel size must be odd.");
    }

    if (options.iterations < 1) {
        throw std::runtime_error("Iterations must be >= 1.");
    }

    return options;
}

float get_pixel(const Image &image, int x, int y) {
    if (x < 0 || y < 0 || x >= image.width || y >= image.height) {
        return 0.0f;
    }
    return image.at(x, y);
}

void convolve_range(const Image &input,
                    const std::vector<float> &kernel,
                    int ksize,
                    int y_start,
                    int y_end,
                    Image &output) {
    int radius = ksize / 2;
    for (int y = y_start; y < y_end; ++y) {
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
}

Image convolve_sequential(const Image &input,
                          const std::vector<float> &kernel,
                          int ksize) {
    Image output;
    output.width = input.width;
    output.height = input.height;
    output.data.assign(input.width * input.height, 0.0f);

    convolve_range(input, kernel, ksize, 0, input.height, output);

    return output;
}

Image sobel_magnitude_sequential(const Image &input) {
    auto kx = make_sobel_x_kernel();
    auto ky = make_sobel_y_kernel();

    Image gx = convolve_sequential(input, kx, 3);
    Image gy = convolve_sequential(input, ky, 3);

    Image output;
    output.width = input.width;
    output.height = input.height;
    output.data.resize(input.width * input.height);

    for (int i = 0; i < input.width * input.height; ++i) {
        float mag = std::sqrt(gx.data[i] * gx.data[i] + gy.data[i] * gy.data[i]);
        output.data[i] = mag;
    }

    return output;
}
