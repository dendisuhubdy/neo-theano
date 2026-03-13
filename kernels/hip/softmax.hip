// Theano CUDA softmax kernel
// Numerically stable: subtract max before exp

#include "../compat.h"

THEANO_DEVICE_FUNC float warp_reduce_sum_sm(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

THEANO_DEVICE_FUNC float warp_reduce_max_sm(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Softmax over the last dimension
// input/output: [outer_size, dim_size]
extern "C" THEANO_GLOBAL_FUNC void softmax_f32(
    const float* input, float* output,
    int outer_size, int dim_size
) {
    int row = THEANO_BLOCK_IDX_X;
    if (row >= outer_size) return;

    int tid = THEANO_THREAD_IDX_X;
    const float* in_row = input + row * dim_size;
    float* out_row = output + row * dim_size;

    // Phase 1: Find max (for numerical stability)
    float max_val = -INFINITY;
    for (int i = tid; i < dim_size; i += THEANO_BLOCK_DIM_X) {
        max_val = fmaxf(max_val, in_row[i]);
    }

    // Reduce max across the block
    THEANO_SHARED float shared_max[32];
    max_val = warp_reduce_max_sm(max_val);
    int lane = tid % 32;
    int warp_id = tid / 32;
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();

    int num_warps = (THEANO_BLOCK_DIM_X + 31) / 32;
    if (warp_id == 0) {
        max_val = (lane < num_warps) ? shared_max[lane] : -INFINITY;
        max_val = warp_reduce_max_sm(max_val);
    }
    // Broadcast max to all threads
    THEANO_SHARED float block_max;
    if (tid == 0) block_max = max_val;
    __syncthreads();
    max_val = block_max;

    // Phase 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = tid; i < dim_size; i += THEANO_BLOCK_DIM_X) {
        float val = expf(in_row[i] - max_val);
        out_row[i] = val;
        sum += val;
    }

    // Reduce sum across the block
    THEANO_SHARED float shared_sum[32];
    sum = warp_reduce_sum_sm(sum);
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        sum = warp_reduce_sum_sm(sum);
    }
    THEANO_SHARED float block_sum;
    if (tid == 0) block_sum = sum;
    __syncthreads();
    sum = block_sum;

    // Phase 3: Normalize
    float inv_sum = 1.0f / sum;
    for (int i = tid; i < dim_size; i += THEANO_BLOCK_DIM_X) {
        out_row[i] *= inv_sum;
    }
}
