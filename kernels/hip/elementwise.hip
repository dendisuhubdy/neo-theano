// Theano CUDA elementwise kernels
// Compatible with both CUDA and HIP via compat.h

#include "../compat.h"

// ---- Unary kernels ----

extern "C" THEANO_GLOBAL_FUNC void neg_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = -input[i];
    }
}

extern "C" THEANO_GLOBAL_FUNC void abs_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

extern "C" THEANO_GLOBAL_FUNC void exp_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = expf(input[i]);
    }
}

extern "C" THEANO_GLOBAL_FUNC void log_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = logf(input[i]);
    }
}

extern "C" THEANO_GLOBAL_FUNC void sqrt_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = sqrtf(input[i]);
    }
}

extern "C" THEANO_GLOBAL_FUNC void sin_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = sinf(input[i]);
    }
}

extern "C" THEANO_GLOBAL_FUNC void cos_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = cosf(input[i]);
    }
}

extern "C" THEANO_GLOBAL_FUNC void tanh_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

extern "C" THEANO_GLOBAL_FUNC void sigmoid_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

extern "C" THEANO_GLOBAL_FUNC void relu_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}

extern "C" THEANO_GLOBAL_FUNC void gelu_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    const float c = 0.7978845608f; // sqrt(2/pi)
    for (int i = idx; i < n; i += stride) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
    }
}

extern "C" THEANO_GLOBAL_FUNC void silu_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

extern "C" THEANO_GLOBAL_FUNC void square_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = input[i] * input[i];
    }
}

extern "C" THEANO_GLOBAL_FUNC void reciprocal_f32(const float* input, float* output, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = 1.0f / input[i];
    }
}

// ---- Binary kernels ----

extern "C" THEANO_GLOBAL_FUNC void add_f32(
    const float* a, const float* b, float* out, int n
) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

extern "C" THEANO_GLOBAL_FUNC void sub_f32(
    const float* a, const float* b, float* out, int n
) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] - b[i];
    }
}

extern "C" THEANO_GLOBAL_FUNC void mul_f32(
    const float* a, const float* b, float* out, int n
) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] * b[i];
    }
}

extern "C" THEANO_GLOBAL_FUNC void div_f32(
    const float* a, const float* b, float* out, int n
) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] / b[i];
    }
}

// ---- Scalar operations ----

extern "C" THEANO_GLOBAL_FUNC void add_scalar_f32(
    const float* input, float scalar, float* output, int n
) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = input[i] + scalar;
    }
}

extern "C" THEANO_GLOBAL_FUNC void mul_scalar_f32(
    const float* input, float scalar, float* output, int n
) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = input[i] * scalar;
    }
}

// ---- Fill ----

extern "C" THEANO_GLOBAL_FUNC void fill_f32(float* output, float value, int n) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = value;
    }
}

// ---- Type casting ----

extern "C" THEANO_GLOBAL_FUNC void cast_f32_to_f64(
    const float* input, double* output, int n
) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = (double)input[i];
    }
}

extern "C" THEANO_GLOBAL_FUNC void cast_f64_to_f32(
    const double* input, float* output, int n
) {
    int idx = THEANO_GLOBAL_IDX;
    int stride = THEANO_GLOBAL_STRIDE;
    for (int i = idx; i < n; i += stride) {
        output[i] = (float)input[i];
    }
}
