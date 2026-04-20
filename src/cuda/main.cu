#include "convolution.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

__global__ void convolve_kernel(const float *input, float *output,
                                int width, int height,
                                const float *kernel, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int radius = ksize / 2;
    float sum = 0.0f;
    for (int ky = 0; ky < ksize; ++ky) {
        for (int kx = 0; kx < ksize; ++kx) {
            int ix = x + kx - radius;
            int iy = y + ky - radius;
            float pixel = 0.0f;
            if (ix >= 0 && iy >= 0 && ix < width && iy < height) {
                pixel = input[iy * width + ix];
            }
            float kval = kernel[ky * ksize + kx];
            sum += pixel * kval;
        }
    }

    output[y * width + x] = sum;
}

__global__ void sobel_magnitude_kernel(const float *gx, const float *gy,
                                       float *output, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    float sx = gx[idx];
    float sy = gy[idx];
    output[idx] = sqrtf(sx * sx + sy * sy);
}

void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

int main(int argc, char **argv) {
    try {
        Options options = parse_args(argc, argv);

        Image input = options.input_path.empty()
                          ? generate_image(options.width, options.height)
                          : load_pgm(options.input_path);

        int width = input.width;
        int height = input.height;
        size_t bytes = width * height * sizeof(float);

        float *d_input = nullptr;
        float *d_output = nullptr;
        check_cuda(cudaMalloc(&d_input, bytes), "cudaMalloc input");
        check_cuda(cudaMalloc(&d_output, bytes), "cudaMalloc output");
        check_cuda(cudaMemcpy(d_input, input.data.data(), bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy input");

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        auto start = std::chrono::high_resolution_clock::now();

        Image output;
        output.width = width;
        output.height = height;
        output.data.resize(width * height);

        if (options.filter == "sobel") {
            auto kx_host = make_sobel_x_kernel();
            auto ky_host = make_sobel_y_kernel();
            float *d_kx = nullptr;
            float *d_ky = nullptr;
            float *d_gx = nullptr;
            float *d_gy = nullptr;

            size_t kbytes = kx_host.size() * sizeof(float);
            check_cuda(cudaMalloc(&d_kx, kbytes), "cudaMalloc kx");
            check_cuda(cudaMalloc(&d_ky, kbytes), "cudaMalloc ky");
            check_cuda(cudaMalloc(&d_gx, bytes), "cudaMalloc gx");
            check_cuda(cudaMalloc(&d_gy, bytes), "cudaMalloc gy");

            check_cuda(cudaMemcpy(d_kx, kx_host.data(), kbytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy kx");
            check_cuda(cudaMemcpy(d_ky, ky_host.data(), kbytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy ky");

            convolve_kernel<<<grid, block>>>(d_input, d_gx, width, height, d_kx, 3);
            convolve_kernel<<<grid, block>>>(d_input, d_gy, width, height, d_ky, 3);

            int total = width * height;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            sobel_magnitude_kernel<<<blocks, threads>>>(d_gx, d_gy, d_output, total);

            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize sobel");

            check_cuda(cudaMemcpy(output.data.data(), d_output, bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy output");

            cudaFree(d_kx);
            cudaFree(d_ky);
            cudaFree(d_gx);
            cudaFree(d_gy);
        } else {
            auto kernel_host = make_blur_kernel(options.kernel_size);
            float *d_kernel = nullptr;
            size_t kbytes = kernel_host.size() * sizeof(float);
            check_cuda(cudaMalloc(&d_kernel, kbytes), "cudaMalloc kernel");
            check_cuda(cudaMemcpy(d_kernel, kernel_host.data(), kbytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy kernel");

            convolve_kernel<<<grid, block>>>(d_input, d_output, width, height, d_kernel, options.kernel_size);
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize blur");

            check_cuda(cudaMemcpy(output.data.data(), d_output, bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy output");
            cudaFree(d_kernel);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (!options.no_output) {
            if (options.output_path.empty()) {
                options.output_path = "results/images/cuda_output.pgm";
            }
            save_pgm(output, options.output_path);
        }

        std::cout << "cuda_elapsed_seconds," << elapsed.count() << "\n";

        cudaFree(d_input);
        cudaFree(d_output);

        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
