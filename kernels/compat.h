// CUDA/HIP Compatibility Header
// This header provides macros and type aliases so that .cu kernel files
// can be compiled for both NVIDIA CUDA and AMD HIP without modification.

#ifndef THEANO_COMPAT_H
#define THEANO_COMPAT_H

#ifdef __HIP_PLATFORM_AMD__
    // HIP is already compatible with most CUDA syntax.
    // Add any HIP-specific overrides here.
    #include <hip/hip_runtime.h>
    #define THEANO_DEVICE_FUNC __device__
    #define THEANO_GLOBAL_FUNC __global__
    #define THEANO_HOST_FUNC __host__
    #define THEANO_HOST_DEVICE __host__ __device__
#else
    // CUDA
    #define THEANO_DEVICE_FUNC __device__
    #define THEANO_GLOBAL_FUNC __global__
    #define THEANO_HOST_FUNC __host__
    #define THEANO_HOST_DEVICE __host__ __device__
#endif

// Shared memory
#define THEANO_SHARED __shared__

// Thread indexing helpers
#define THEANO_THREAD_IDX_X threadIdx.x
#define THEANO_THREAD_IDX_Y threadIdx.y
#define THEANO_BLOCK_IDX_X blockIdx.x
#define THEANO_BLOCK_IDX_Y blockIdx.y
#define THEANO_BLOCK_DIM_X blockDim.x
#define THEANO_BLOCK_DIM_Y blockDim.y
#define THEANO_GRID_DIM_X gridDim.x
#define THEANO_GRID_DIM_Y gridDim.y

// Global thread index (1D)
#define THEANO_GLOBAL_IDX (THEANO_BLOCK_IDX_X * THEANO_BLOCK_DIM_X + THEANO_THREAD_IDX_X)
#define THEANO_GLOBAL_STRIDE (THEANO_GRID_DIM_X * THEANO_BLOCK_DIM_X)

#endif // THEANO_COMPAT_H
