# HPC Case Study

High Performance Computing and Parallel Programming
                          (22CP702T)
                                         Case Study
Image filtering is a fundamental operation in digital image processing, widely used in applications
such as noise reduction, sharpening, and feature extraction. The operation is typically implemented
using Convolution, where each output pixel is computed as a weighted sum of its neighboring
pixels. For high-resolution images (e.g., 4K or larger), convolution becomes computationally
expensive when executed sequentially. This creates a need for efficient parallel implementations
using modern multi-core and distributed computing systems.
 Design and implement a parallel version of 2D image convolution and evaluate its
  performance against a sequential implementation.

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
