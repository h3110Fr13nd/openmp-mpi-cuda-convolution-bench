# Image Filtering with Parallel Convolution - HPC Case Study

## Introduction



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


---

All match till now, is it actually useful or helpful. Still why this is important, or what can it actually do.

Let's take an example of blur filter, which is a common application of convolution in image processing.