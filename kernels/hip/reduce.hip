// Theano CUDA reduction kernels

#include "../compat.h"

// Warp-level reduction using shuffle
THEANO_DEVICE_FUNC float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

THEANO_DEVICE_FUNC float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction
extern "C" THEANO_GLOBAL_FUNC void reduce_sum_f32(
    const float* input, float* output, int n
) {
    THEANO_SHARED float shared[32]; // one per warp

    int tid = THEANO_THREAD_IDX_X;
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;

    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces across warps
    int num_warps = (THEANO_BLOCK_DIM_X + 31) / 32;
    if (warp_id == 0) {
        sum = (lane < num_warps) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            atomicAdd(output, sum);
        }
    }
}

extern "C" THEANO_GLOBAL_FUNC void reduce_max_f32(
    const float* input, float* output, int n
) {
    THEANO_SHARED float shared[32];

    int tid = THEANO_THREAD_IDX_X;
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;

    float max_val = -INFINITY;
    for (int i = idx; i < n; i += stride) {
        max_val = fmaxf(max_val, input[i]);
    }

    max_val = warp_reduce_max(max_val);

    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    int num_warps = (THEANO_BLOCK_DIM_X + 31) / 32;
    if (warp_id == 0) {
        max_val = (lane < num_warps) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max(max_val);
        if (lane == 0) {
            // atomicMax for float — use CAS loop
            int* addr_as_int = (int*)output;
            int old = *addr_as_int;
            int expected;
            do {
                expected = old;
                float old_f = __int_as_float(old);
                float new_f = fmaxf(old_f, max_val);
                old = atomicCAS(addr_as_int, expected, __float_as_int(new_f));
            } while (old != expected);
        }
    }
}
