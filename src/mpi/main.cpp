#include "convolution.hpp"

#include <chrono>
#include <cmath>
#include <iostream>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace {
void partition_rows(int total_rows, int rank, int size, int &start, int &count) {
    int base = total_rows / size;
    int rem = total_rows % size;
    count = base + (rank < rem ? 1 : 0);
    start = rank * base + std::min(rank, rem);
}

Image convolve_subset(const Image &input,
                      const std::vector<float> &kernel,
                      int ksize,
                      int y_start,
                      int y_end) {
    Image output;
    output.width = input.width;
    output.height = y_end - y_start;
    output.data.assign(output.width * output.height, 0.0f);

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
            output.at(x, y - y_start) = sum;
        }
    }

    return output;
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

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<float> kernel;
        int ksize = options.kernel_size;
        if (options.filter == "sobel") {
            ksize = 3;
            auto kx = make_sobel_x_kernel();
            auto ky = make_sobel_y_kernel();

            int start_row = 0;
            int row_count = 0;
            partition_rows(height, rank, size, start_row, row_count);
            Image gx = convolve_subset(input, kx, 3, start_row, start_row + row_count);
            Image gy = convolve_subset(input, ky, 3, start_row, start_row + row_count);

            Image local_out;
            local_out.width = width;
            local_out.height = row_count;
            local_out.data.resize(width * row_count);
            for (int i = 0; i < width * row_count; ++i) {
                float mag = std::sqrt(gx.data[i] * gx.data[i] + gy.data[i] * gy.data[i]);
                local_out.data[i] = mag;
            }

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

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;

            if (rank == 0 && !options.no_output) {
                if (options.output_path.empty()) {
                    options.output_path = "results/images/mpi_output.pgm";
                }
                save_pgm(output, options.output_path);
            }

            if (rank == 0) {
                std::cout << "mpi_ranks," << size << "\n";
                std::cout << "mpi_elapsed_seconds," << elapsed.count() << "\n";
            }

            MPI_Finalize();
            return 0;
        }

        kernel = make_blur_kernel(ksize);
        int start_row = 0;
        int row_count = 0;
        partition_rows(height, rank, size, start_row, row_count);

        Image local_out = convolve_subset(input, kernel, ksize, start_row, start_row + row_count);

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

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        if (rank == 0 && !options.no_output) {
            if (options.output_path.empty()) {
                options.output_path = "results/images/mpi_output.pgm";
            }
            save_pgm(output, options.output_path);
        }

        if (rank == 0) {
            std::cout << "mpi_ranks," << size << "\n";
            std::cout << "mpi_elapsed_seconds," << elapsed.count() << "\n";
        }

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
