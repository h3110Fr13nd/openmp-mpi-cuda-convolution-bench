#include "convolution.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace {
void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

void partition_rows(int total_rows, int rank, int size, int &start, int &count) {
    int base = total_rows / size;
    int rem = total_rows % size;
    count = base + (rank < rem ? 1 : 0);
    start = rank * base + std::min(rank, rem);
}

__global__ void convolve_subset_kernel(const float *input, float *output,
                                       int width, int height,
                                       const float *kernel, int ksize,
                                       int y_start, int y_end) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + y_start;

    if (x >= width || y >= y_end) {
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

    int local_y = y - y_start;
    output[local_y * width + x] = sum;
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
} // namespace

int main(int argc, char **argv) {
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        Options options = parse_args(argc, argv);

        Image input;
        if (rank == 0) {
            input = options.input_path.empty()
                        ? generate_image(options.width, options.height)
                        : load_pgm(options.input_path);
        }

        int width = input.width;
        int height = input.height;
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            input.width = width;
            input.height = height;
            input.data.resize(width * height);
        }

        MPI_Bcast(input.data.data(), width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);

        int start_row = 0;
        int row_count = 0;
        partition_rows(height, rank, size, start_row, row_count);

        size_t full_bytes = width * height * sizeof(float);
        size_t local_bytes = width * row_count * sizeof(float);

        float *d_input = nullptr;
        float *d_out = nullptr;
        check_cuda(cudaMalloc(&d_input, full_bytes), "cudaMalloc input");
        check_cuda(cudaMalloc(&d_out, local_bytes), "cudaMalloc output");
        check_cuda(cudaMemcpy(d_input, input.data.data(), full_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy input");

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                  (row_count + block.y - 1) / block.y);

        auto start_time = std::chrono::high_resolution_clock::now();

        Image local_out;
        local_out.width = width;
        local_out.height = row_count;
        local_out.data.resize(width * row_count);

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
            check_cuda(cudaMalloc(&d_gx, local_bytes), "cudaMalloc gx");
            check_cuda(cudaMalloc(&d_gy, local_bytes), "cudaMalloc gy");
            check_cuda(cudaMemcpy(d_kx, kx_host.data(), kbytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy kx");
            check_cuda(cudaMemcpy(d_ky, ky_host.data(), kbytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy ky");

            convolve_subset_kernel<<<grid, block>>>(d_input, d_gx, width, height, d_kx, 3, start_row, start_row + row_count);
            convolve_subset_kernel<<<grid, block>>>(d_input, d_gy, width, height, d_ky, 3, start_row, start_row + row_count);

            int total = width * row_count;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            sobel_magnitude_kernel<<<blocks, threads>>>(d_gx, d_gy, d_out, total);

            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize sobel");

            check_cuda(cudaMemcpy(local_out.data.data(), d_out, local_bytes, cudaMemcpyDeviceToHost),
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

            convolve_subset_kernel<<<grid, block>>>(d_input, d_out, width, height, d_kernel, options.kernel_size, start_row, start_row + row_count);
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize blur");

            check_cuda(cudaMemcpy(local_out.data.data(), d_out, local_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy output");
            cudaFree(d_kernel);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::vector<int> recv_counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            int s = 0;
            int c = 0;
            partition_rows(height, r, size, s, c);
            recv_counts[r] = c * width;
            displs[r] = s * width;
        }

        Image output;
        if (rank == 0) {
            output.width = width;
            output.height = height;
            output.data.resize(width * height);
        }

        MPI_Gatherv(local_out.data.data(), row_count * width, MPI_FLOAT,
                    output.data.data(), recv_counts.data(), displs.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);

        if (rank == 0 && !options.no_output) {
            if (options.output_path.empty()) {
                options.output_path = "results/images/mpi_cuda_output.pgm";
            }
            save_pgm(output, options.output_path);
        }

        if (rank == 0) {
            std::cout << "mpi_cuda_ranks," << size << "\n";
            std::cout << "mpi_cuda_elapsed_seconds," << elapsed.count() << "\n";
        }

        cudaFree(d_input);
        cudaFree(d_out);

        MPI_Finalize();
        return 0;
    } catch (const std::exception &ex) {
        if (rank == 0) {
            std::cerr << "Error: " << ex.what() << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }
#else
    std::cerr << "MPI support not enabled.\n";
    return 1;
#endif
}
