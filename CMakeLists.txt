cmake_minimum_required(VERSION 3.20)
project(hpc_image_filtering LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(OPENMP_TARGET_NVPTX "Enable OpenMP target offload to NVIDIA GPUs" OFF)
set(OPENMP_NVPTX_BC_PATH "" CACHE FILEPATH "Path to libomptarget NVPTX bitcode (e.g., libomptarget-nvptx-sm_86.bc)")

add_library(conv_common
	src/common/convolution.cpp
)

target_include_directories(conv_common
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(seq_convolution src/sequential/main.cpp)
target_link_libraries(seq_convolution PRIVATE conv_common)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	add_executable(omp_convolution src/openmp/main.cpp)
	target_link_libraries(omp_convolution PRIVATE conv_common OpenMP::OpenMP_CXX)
	target_compile_definitions(omp_convolution PRIVATE HAVE_OPENMP=1)

	add_executable(omp_target_convolution src/openmp/target_main.cpp)
	target_link_libraries(omp_target_convolution PRIVATE conv_common OpenMP::OpenMP_CXX)
	target_compile_definitions(omp_target_convolution PRIVATE HAVE_OPENMP=1)

	if(OPENMP_TARGET_NVPTX)
		if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
			find_package(CUDAToolkit)
			if(CUDAToolkit_FOUND)
				if(NOT OPENMP_NVPTX_BC_PATH)
					foreach(candidate
						"/usr/lib/llvm-21/lib/nvptx64-nvidia-cuda/libomptarget-nvptx.bc"
						"/usr/lib/llvm-17/lib/libomptarget-nvptx-sm_86.bc"
						"/usr/lib/llvm-17/lib/libomptarget-nvptx-sm_80.bc"
						"/usr/lib/llvm-18/lib/libomptarget-nvptx-sm_86.bc"
						"/usr/lib/llvm-18/lib/libomptarget-nvptx-sm_80.bc"
						"/usr/lib/llvm-19/lib/libomptarget-nvptx-sm_86.bc"
						"/usr/lib/llvm-19/lib/libomptarget-nvptx-sm_80.bc"
					)
						if(EXISTS ${candidate})
							set(OPENMP_NVPTX_BC_PATH ${candidate})
							break()
						endif()
					endforeach()
				endif()

				if(NOT OPENMP_NVPTX_BC_PATH)
					message(WARNING "OpenMP NVPTX bitcode not found. Set OPENMP_NVPTX_BC_PATH to libomptarget-nvptx-sm_XX.bc for your GPU.")
				else()
				target_compile_options(omp_target_convolution PRIVATE
					-fopenmp
					-fopenmp-targets=nvptx64-nvidia-cuda
						-Xopenmp-target=nvptx64-nvidia-cuda --cuda-path=${CUDAToolkit_ROOT}
						-Xopenmp-target=nvptx64-nvidia-cuda --libomptarget-nvptx-bc-path=${OPENMP_NVPTX_BC_PATH}
				)
				target_link_options(omp_target_convolution PRIVATE
					-fopenmp
					-fopenmp-targets=nvptx64-nvidia-cuda
						-Xopenmp-target=nvptx64-nvidia-cuda --cuda-path=${CUDAToolkit_ROOT}
						-Xopenmp-target=nvptx64-nvidia-cuda --libomptarget-nvptx-bc-path=${OPENMP_NVPTX_BC_PATH}
				)
				target_compile_definitions(omp_target_convolution PRIVATE OPENMP_TARGET_NVPTX=1)
				endif()
			else()
				message(WARNING "CUDAToolkit not found; OpenMP target offload disabled.")
			endif()
		else()
			message(WARNING "OpenMP target offload requires Clang; disable OPENMP_TARGET_NVPTX or reconfigure with Clang.")
		endif()
	endif()
endif()

find_package(MPI)
if(MPI_CXX_FOUND)
	add_executable(mpi_convolution src/mpi/main.cpp)
	target_link_libraries(mpi_convolution PRIVATE conv_common MPI::MPI_CXX)
	target_compile_definitions(mpi_convolution PRIVATE HAVE_MPI=1)
endif()

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
	enable_language(CUDA)
	add_executable(cuda_convolution src/cuda/main.cu)
	set_target_properties(cuda_convolution PROPERTIES
		CUDA_STANDARD 17
		CUDA_STANDARD_REQUIRED ON
	)
	target_include_directories(cuda_convolution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
	target_link_libraries(cuda_convolution PRIVATE conv_common CUDA::cudart)
	target_compile_definitions(cuda_convolution PRIVATE HAVE_CUDA=1)

	if(MPI_CXX_FOUND)
		add_executable(mpi_cuda_convolution src/mpi/cuda_main.cu)
		set_target_properties(mpi_cuda_convolution PROPERTIES
			CUDA_STANDARD 17
			CUDA_STANDARD_REQUIRED ON
		)
		target_include_directories(mpi_cuda_convolution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
		target_link_libraries(mpi_cuda_convolution PRIVATE conv_common CUDA::cudart MPI::MPI_CXX)
		target_compile_definitions(mpi_cuda_convolution PRIVATE HAVE_MPI=1 HAVE_CUDA=1)
	endif()
endif()
