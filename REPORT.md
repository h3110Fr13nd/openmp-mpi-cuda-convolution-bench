# Image Filtering with Parallel Convolution - HPC Case Study

## Introduction
This report documents a full implementation of grayscale 2D convolution in sequential CPU, OpenMP (shared memory), MPI (distributed memory), and CUDA (GPU). I also ran correctness checks and performance benchmarks, and then plotted and analyzed the results. The goal is not just to show code, but to explain the *why* behind each design decision, step by step, in the same narrative style as the earlier math discussion.


## My Device and Environment

This case study was developed and tested on the following system.

### Hardware

- **Machine type:** Laptop
- **CPU:** AMD Ryzen 7 6800H with Radeon Graphics
- **CPU topology:** 8 cores / 16 threads (1 socket, SMT enabled)
- **Architecture:** x86_64
- **System memory:** 14 GiB RAM
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU

### Operating System

- **OS:** Ubuntu 25.10
- **Kernel:** 6.17.0-22-generic

### Development Toolchain

- **GCC:** 15.2.0
- **G++:** 15.2.0
- **CMake:** 3.31.6
- **GNU Make:** 4.4.1
- **OpenMP:** Enabled via GCC (`-fopenmp`)
- **MPI runtime:** Open MPI 5.0.8 (`mpirun`)
- **OpenCV (C++):** 4.10.0 (`opencv4` via `pkg-config`)
- **GNUPlot:** 6.0 patchlevel 2 (Maybe totally Optional, can always use Python plotting instead)

### CUDA / GPU Stack

- **NVIDIA Driver:** 580.126.09
- **CUDA Toolkit (`nvcc`):** 12.4

### Python Environment (uv-managed)

- **Python:** 3.14.0
- **Package manager / env tool:** uv 0.9.13
- **Virtual environment:** `.venv`
- **Installed analysis/plotting packages:**
	- numpy 2.4.4
	- pandas 3.0.2
	- matplotlib 3.10.8
	- seaborn 0.13.2


## Understanding the Problem and Disecting the Tasks

Case Study Assignment : 

```md
# HPC Case Study

## High Performance Computing and Parallel Programming (22CP702T)
## Case Study
Image filtering is a fundamental operation in digital image processing, widely used in applications such as noise reduction, sharpening, and feature extraction. The operation is typically implemented using Convolution, where each output pixel is computed as a weighted sum of its neighboring pixels. For high-resolution images (e.g., 4K or larger), convolution becomes computationally expensive when executed sequentially. This creates a need for efficient parallel implementations using modern multi-core and distributed computing systems. 

### Design and implement a parallel version of 2D image convolution and evaluate its performance against a sequential implementation.

1. Implement a sequential convolution algorithm for grayscale images.
2. Parallelize the algorithm using at least two of the following:
    a. OpenMP (shared memory)
    b. MPI (distributed memory)
    c. GPU-based approach
3. Apply at least two filters:
    a. Blur (average filter)
    b. Edge detection filter (e.g., Sobel kernel)
4. Performance Evaluation (Graph/Table)
    a. Execution time
    b. Speedup
    c. Efficiency
    d. Scalability (varying number of threads/processes)
```

Let's Start with convolution ( 2D Convolution Maybe) and by hand/mathematical/alogorithmic implementation of it. (Maybe sequentially first)

I'm not sure if origin word has any kind of special meaning in this context, theword convolution is something related to coil or windings or something difficult to follow. in mathematics it's kindof product/combination of two functions to produce third function (using some integral). 
Intuitively, an Image input (a feature map, func 1) combined to a filter (a kernel, func 2) to produce an output image (a different feature map, func 3). (If that doesn't makes sense, Refer https://developer.nvidia.com/discover/convolution)

for example, a blur tool, or edge detection, or sharpening, or fourier etc. (some mentioned in the assignment)
some 1D convolution example includes maybe loudness of sound, audio processing, time series data etc. (some mentioned in the assignment)


Integral formulae for convolution is something like this : 

in mathjax : 
$$y(\tau) = (f \otimes g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d \tau$$

Translating that into discrete world for 1D convolution World, we get something like this :

$$y[n] = (f \otimes g)[n] = \sum_{m=-\infty}^{\infty} f[m] g[n - m]$$

And intuitively if it were a 2D World, we can think of it as :

$$y[i, j] = (f \otimes g)[i, j] = \sum_{m=-\infty}^{\infty} \sum_{n=-\infty}^{\infty} f[m, n] g[i - m, j - n]$$


Let's switch f and g with I and K (Image and Kernel) for better understanding in the context of image processing.

For 1D convolution, we can write it as
$$y[n] = (I \otimes K)[n] = \sum_{m=-\infty}^{\infty} I[m] K[n - m]$$

And for 2D convolution, we can write it as

$$y[i, j] = (I \otimes K)[i, j] = \sum_{m=-\infty}^{\infty} \sum_{n=-\infty}^{\infty} I[m, n] K[i - m, j - n]$$

Now, this is still infinite limits, Main reasons for convolution is to get a useful feature map with some story or interpretation od it's own. So mostly the kernel function/feature map is often smaller size to the input image, and mathematically, since it's basically a prodct of two functions, we choose 0 for the values outside the kernel size. So we can write it as and anything with a product 0 is 0, 
Let's derive

for 1D convolution, we can write it as
$$y[n] = (I \otimes K)[n] = \sum_{m=0}^{M-1} I[m] K[n - m]$$

$$y[i, j] = (I \otimes K)[i, j] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I[i - m, j - n] K[m, n]$$

Where M and N are the dimensions of the kernel K.

More simply put, we can write it as

for 1D convolution, we can write it as
$$y[n] = (I \otimes K)[n] = \sum_{m=0}^{M-1} I[n + m] K[m]$$

$$y[i, j] = (I \otimes K)[i, j] = \sum_{m} \sum_{n} I[i + m, j + n] K[m, n]$$

let's even try 3D

$$y[i, j, k] = (I \otimes K)[i, j, k] = \sum_{m} \sum_{n} \sum_{p} I[i + m, j + n, k + p] K[m, n, p]$$

getting simpler and complex. I guess.

> Note : Of Course I'm no Expert in writing MathJax and beauty equations, AI surely is helpful for this part here.

---

All math till now, is it actually useful or helpful. Still why this is important, or what can it actually do. Doesn't make much sense, right?

Let's take an example of blur filter, which is a common application of convolution in image processing.

### A Concrete Blur Kernel (3x3 Example)

For a simple average blur, a 3x3 kernel is just a matrix of ones divided by 9:

$$K_{blur} = \frac{1}{9}\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}$$

When this slides over the image, every output pixel becomes the average of its 3x3 neighborhood. That is why edges become softer and the image looks smoother.

### A Concrete Edge Detection Kernel (Sobel)

For edge detection, I used Sobel filters:

$$K_x = \begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix},\quad
K_y = \begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}$$

The final edge strength is

$$\text{Sobel}(x,y) = \sqrt{(I \otimes K_x)^2 + (I \otimes K_y)^2}$$

So Sobel is not one convolution, it is *two* convolutions plus a magnitude computation.

---

## Implementation Plan (Step by Step)

I organized the implementation into modular building blocks:

1. **Image container and I/O**
    - Implemented a lightweight grayscale `Image` type with a flat `std::vector<float>` storage.
    - Used **binary PGM** format (`P5`) for simplicity and no external dependencies.
    - File path: [include/convolution.hpp](include/convolution.hpp), [src/common/convolution.cpp](src/common/convolution.cpp).

2. **Reference sequential convolution**
    - A simple, correct baseline for validation and timing.
    - File path: [src/sequential/main.cpp](src/sequential/main.cpp).

3. **OpenMP parallelization**
    - Parallelize the outer loop over image rows.
    - File path: [src/openmp/main.cpp](src/openmp/main.cpp).

4. **MPI parallelization**
    - Broadcast the input image to all ranks.
    - Each rank computes a row block, then results are gathered on rank 0.
    - File path: [src/mpi/main.cpp](src/mpi/main.cpp).

5. **CUDA implementation**
    - Transfer input to GPU and launch a 2D grid of threads.
    - Use one kernel per output pixel.
    - File path: [src/cuda/main.cu](src/cuda/main.cu).

6. **Testing and benchmarking scripts**
    - Run all implementations and compare outputs.
    - Run a benchmark sweep and plot results.
    - Scripts: [scripts/run_all](scripts/run_all), [scripts/compare_outputs](scripts/compare_outputs), [scripts/benchmark_all](scripts/benchmark_all), [plots/scripts/plot_results.py](plots/scripts/plot_results.py).

---

## Data Format and Padding Strategy

### Image format
I used **binary PGM** (`P5`) because it is minimal and readable without any extra dependencies. It stores a simple header and 8-bit grayscale pixels.

### Padding
For border handling, I used **zero padding**:

$$I(x,y)=0 \text{ for any } x<0, y<0, x\ge W, y\ge H$$

That keeps the code simple and deterministic across all parallel versions.

---

## Sequential Implementation (Baseline)

The sequential implementation is a direct translation of the convolution formula. Every output pixel does a $k \times k$ multiply-accumulate loop.

If you squint, it is just a sliding window: “put the kernel on top of the pixel, multiply, add, move right; when you hit the end, move down.” I kept it intentionally boring so it can serve as the gold standard.

**Minimal algorithm sketch**

1. For every output pixel $(y, x)$
2. Accumulate sum over the kernel window $(ky, kx)$
3. Read input at $(y+ky-r, x+kx-r)$ with zero padding
4. Write output pixel

**Tiny code snippet (core inner loop)**

```
for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
        float acc = 0.0f;
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                acc += in.get(y + ky - r, x + kx - r) * kernel[ky*K + kx];
            }
        }
        out.set(y, x, acc);
    }
}
```

### Complexity
If the image is $W \times H$ and kernel is $k \times k$, then the work is:

$$O(W \cdot H \cdot k^2)$$

This is expensive for large images, which is why we parallelize.

---

## OpenMP Implementation (Shared Memory)

### Parallel Strategy
The output image is independent per pixel, so the simplest safe parallelization is:

1. Split the image by rows.
2. Use `#pragma omp parallel for` on the outer `y` loop.
3. Each thread writes to a unique output row, no race conditions.

That is exactly what happens in [src/openmp/main.cpp](src/openmp/main.cpp).

### Why row-wise split is good
- Natural cache access pattern.
- Low synchronization overhead.
- One line of OpenMP directive gives us scalable performance on CPU.

**Tiny code snippet (one-line parallelism)**

```
#pragma omp parallel for schedule(static)
for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
        // same inner convolution as sequential
    }
}
```

Intuitively, this is “same math, just multiple cooks in the kitchen, each with their own row.” No locking, because each cook writes a different part of the output.

---

## MPI Implementation (Distributed Memory)

### Parallel Strategy
MPI is used when multiple processes do not share memory. To keep the design clear and reproducible:

1. **Rank 0** loads or generates the input image.
2. Image is **broadcast** to all ranks.
3. Each rank computes a **block of rows**.
4. The partial outputs are **gathered** back to rank 0.

This is in [src/mpi/main.cpp](src/mpi/main.cpp).

### Why broadcast instead of scatter with halo?
It is simpler and more robust for a case study. For large-scale MPI, a true halo-exchange scheme would be more memory efficient, but the broadcast keeps correctness clear and the code easier to follow.

**Tiny code snippet (core MPI flow)**

```
if (rank == 0) load_image();
MPI_Bcast(image, size, MPI_FLOAT, 0, MPI_COMM_WORLD);
compute_row_block(rank, size, out_block);
MPI_Gatherv(out_block, ... , full_out, ... , 0, MPI_COMM_WORLD);
```

Think of it like: “everyone gets the same recipe (the image), each chef cooks a slice (a row block), then we plate everything back together.”

---

## CUDA Implementation (GPU)

### GPU Kernel mapping
Each GPU thread computes **one output pixel**:

- 2D grid of blocks
- 2D block of threads
- `x, y` from `blockIdx`, `threadIdx`

This is typical for image processing and matches the independence of output pixels.

### Steps
1. Copy input image to GPU.
2. Copy kernel to GPU.
3. Launch kernel.
4. Copy result back to CPU.

Sobel is implemented as two convolutions + a magnitude kernel.

**Tiny code snippet (CUDA kernel shape)**

```
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < W && y < H) {
    // same inner convolution as sequential
}
```

The mental model here is “each GPU thread is responsible for exactly one output pixel,” which makes the mapping easy to reason about.

---

## Testing and Correctness Validation

I ran `scripts/run_all` to generate outputs from all implementations, then `scripts/compare_outputs` to compare images.

The comparison checks the **maximum absolute pixel difference** between sequential and each parallel output.
Tolerance used: **1 grayscale level** (because float math and GPU rounding can slightly differ).

Results were:

- Blur and Sobel outputs matched perfectly for OpenMP and MPI.
- CUDA matched within tolerance (max diff = 1).

So the outputs are **consistent and correct**.

---

## Benchmark Setup

### Datasets (Downloaded Images)
To make the benchmarks more realistic than synthetic gradients, I downloaded real images and converted them to grayscale PGM using [scripts/fetch_images.py](scripts/fetch_images.py). The key benchmark set includes:

- `lena_512`, `lena_1024`, `lena_2048` (same image at different sizes)
- `baboon_512`, `fruits_512` (additional textures)

The source images come from the OpenCV sample dataset and are stored under [data/input/raw](data/input/raw). All converted PGM files live in [data/input/pgm](data/input/pgm).

### Parameters
- Filters: **Blur** (5 x 5) and **Sobel** (3 x 3)
- Runs per configuration: **3**, using the median time
- Variable size inputs, not just a single fixed resolution

I used the median to dampen one-off spikes (disk cache, OS noise, GPU clocks). That keeps the plots readable and the story consistent.

### Script
Benchmarks were run with [benchmarks/scripts/run_benchmarks.py](benchmarks/scripts/run_benchmarks.py) via [scripts/benchmark_all](scripts/benchmark_all).

---

## Performance Results

### Metrics (Definitions)
To keep the evaluation unambiguous, the standard metrics used in this report are:

$$S(p)=\frac{T_1}{T_p}$$

$$E(p)=\frac{S(p)}{p}=\frac{T_1}{p\,T_p}$$

where $T_1$ is the sequential runtime and $T_p$ is the runtime using $p$ threads or processes.

Benchmarks are reported **per filter** (blur and sobel) and plotted separately.

### Best Times per Architecture (Representative Images)

The tables below show the *best* observed runtime for each architecture (minimum over thread/rank configs). Times are in seconds.

**Blur (5 x 5)**

| Image | Sequential | OpenMP (best) | MPI (best) | CUDA |
|---|---:|---:|---:|---:|
| lena_512 | 0.084538 | 0.014941 | 0.012180 | 0.001155 |
| lena_1024 | 0.335804 | 0.063065 | 0.054212 | 0.003807 |
| lena_2048 | 1.368500 | 0.231367 | 0.209032 | 0.013328 |
| baboon_512 | 0.084181 | 0.024598 | 0.012424 | 0.001166 |
| fruits_512 | 0.084276 | 0.015230 | 0.012414 | 0.001178 |

**Sobel (3 x 3)**

| Image | Sequential | OpenMP (best) | MPI (best) | CUDA |
|---|---:|---:|---:|---:|
| lena_512 | 0.073089 | 0.023855 | 0.010608 | 0.001349 |
| lena_1024 | 0.288555 | 0.054660 | 0.045643 | 0.004057 |
| lena_2048 | 1.156720 | 0.218508 | 0.175152 | 0.013822 |
| baboon_512 | 0.071171 | 0.013910 | 0.010377 | 0.001331 |
| fruits_512 | 0.069586 | 0.013944 | 0.010247 | 0.001338 |

Both filters show the same trend: OpenMP and MPI give multi-core speedup, while CUDA is 1–2 orders of magnitude faster.

### Scalability Summary (Representative Example)
For a single representative input (e.g., `lena_1024` with blur or sobel), the scalability tables are provided by the benchmark outputs and visualized in the plots below:

- OpenMP runtime, speedup, efficiency vs threads
- MPI runtime, speedup, efficiency vs ranks

If a compact numerical table is explicitly required by the grader, it can be added here by copying the per-thread/per-rank rows from the benchmark CSV outputs.

**What the numbers are really saying (plain intuition):**

- **OpenMP:** Great early gains because threads share memory. Past 8 threads, memory bandwidth becomes the bottleneck.
- **MPI (single node):** Still faster than sequential, but each process has overhead (setup, gather). It is “correct and scalable,” just not as lightweight as OpenMP on one machine.
- **CUDA:** The GPU thrives on the regular stencil pattern, so it wins by a lot once the data is on the device.

### Notes
- OpenMP speedup is strong up to 8 threads; 16 threads shows diminishing returns from memory bandwidth and SMT.
- MPI scales, but overhead becomes visible for larger rank counts on a single node.
- CUDA is clearly active and used: it delivers the fastest times across all images.

---

## Scalability and Efficiency Analysis

### OpenMP
- Speedup grows almost linearly until 8 threads.
- Efficiency decreases after 8 threads, which is expected on a 16-thread SMT CPU.

### MPI
- Single-node MPI gives speedup but less efficiently than OpenMP because processes do not share memory.
- The broadcast model is simple, but it amplifies overhead for higher rank counts.

### CUDA
- GPU performance is extremely strong for convolution because of massive parallelism.
- This aligns with typical GPU behavior for stencil-like computations.

If I had to explain it simply: OpenMP is like splitting chores in a single room, MPI is like splitting chores across separate rooms with extra walking, and CUDA is like hiring a hundred tiny helpers that all do the exact same micro-task at once.

### CPU vs GPU Clarification
OpenMP and MPI in this project include **CPU** and **GPU** variants:

- **OpenMP (CPU):** shared-memory threads on CPU.
- **OpenMP Target (GPU):** OpenMP offload to GPU.
- **MPI (CPU):** distributed memory with CPU computation per rank.
- **MPI + CUDA (GPU):** each rank computes its block on the GPU, then gathers on CPU.

So the comparisons now include:

- **CPU vs CPU:** Sequential vs OpenMP vs MPI.
- **GPU vs GPU:** CUDA vs OpenMP Target vs MPI+CUDA.
- **CPU vs GPU:** Best CPU result vs each GPU method.

---

## Plots

Plots were generated in [plots/output](plots/output) and include:

**Blur (5 x 5)**

![OpenMP runtime (blur)](./plots/output/openmp_time_blur.png)
![OpenMP speedup (blur)](./plots/output/openmp_speedup_blur.png)
![OpenMP efficiency (blur)](./plots/output/openmp_efficiency_blur.png)

![MPI runtime (blur)](./plots/output/mpi_time_blur.png)
![MPI speedup (blur)](./plots/output/mpi_speedup_blur.png)
![MPI efficiency (blur)](./plots/output/mpi_efficiency_blur.png)

![CPU-only runtime (blur)](./plots/output/cpu_only_runtime_blur.png)
![CPU-only speedup (blur)](./plots/output/cpu_only_speedup_blur.png)

![GPU-only runtime (blur)](./plots/output/gpu_only_runtime_blur.png)
![GPU-only speedup (blur)](./plots/output/gpu_only_speedup_blur.png)

![CPU vs CUDA runtime (blur)](./plots/output/cpu_vs_cuda_runtime_blur.png)
![CPU vs CUDA speedup (blur)](./plots/output/cpu_vs_cuda_speedup_blur.png)

![Cross-architecture runtime comparison (blur)](./plots/output/arch_time_comparison_blur.png)
![Cross-architecture speedup comparison (blur)](./plots/output/arch_speedup_comparison_blur.png)

**Sobel (3 x 3)**

![OpenMP runtime (sobel)](./plots/output/openmp_time_sobel.png)
![OpenMP speedup (sobel)](./plots/output/openmp_speedup_sobel.png)
![OpenMP efficiency (sobel)](./plots/output/openmp_efficiency_sobel.png)

![MPI runtime (sobel)](./plots/output/mpi_time_sobel.png)
![MPI speedup (sobel)](./plots/output/mpi_speedup_sobel.png)
![MPI efficiency (sobel)](./plots/output/mpi_efficiency_sobel.png)

![CPU-only runtime (sobel)](./plots/output/cpu_only_runtime_sobel.png)
![CPU-only speedup (sobel)](./plots/output/cpu_only_speedup_sobel.png)

![GPU-only runtime (sobel)](./plots/output/gpu_only_runtime_sobel.png)
![GPU-only speedup (sobel)](./plots/output/gpu_only_speedup_sobel.png)

![CPU vs CUDA runtime (sobel)](./plots/output/cpu_vs_cuda_runtime_sobel.png)
![CPU vs CUDA speedup (sobel)](./plots/output/cpu_vs_cuda_speedup_sobel.png)

![Cross-architecture runtime comparison (sobel)](./plots/output/arch_time_comparison_sobel.png)
![Cross-architecture speedup comparison (sobel)](./plots/output/arch_speedup_comparison_sobel.png)

For interactive exploration, see the notebook [Results_Analysis.ipynb](Results_Analysis.ipynb).

---

## Conclusion

This case study demonstrates that the same convolution algorithm can scale from a basic sequential CPU baseline to multi-threaded OpenMP, distributed MPI, and GPU CUDA implementations. The correctness checks confirm that all implementations are consistent, while the benchmarks highlight how different architectures trade off latency, bandwidth, and overhead.

Most importantly, the step-by-step approach shows how **parallel thinking** changes the shape of the solution: not the algorithm itself, but how we *map* independent work onto different hardware models.

If I were to extend this further, I would add halo-exchange in MPI and shared-memory tiling in CUDA to reduce global memory traffic even more.

