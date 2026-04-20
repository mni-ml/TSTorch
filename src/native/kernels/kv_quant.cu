#include <stdint.h>
#include <math.h>

extern "C" __global__
void compute_rowwise_scale_f32(
    const float* __restrict__ input,   // [rows, D]
    float* __restrict__ scales,        // [rows]
    int rows,
    int d
) {
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    float max_abs = 0.0f;
    int row_off = row * d;
    for (int i = 0; i < d; i++) {
        float v = fabsf(input[row_off + i]);
        if (v > max_abs) max_abs = v;
    }
    float scale = fmaxf(max_abs / 127.0f, 1e-8f);
    scales[row] = scale;
}

extern "C" __global__
void quantize_rowwise_i8_f32(
    const float* __restrict__ input,   // [rows, D]
    const float* __restrict__ scales,  // [rows]
    int8_t* __restrict__ output,       // [rows, D]
    int rows,
    int d
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = rows * d;
    if (idx >= total) return;
    int row = idx / d;
    float scale = scales[row];
    float inv_scale = 1.0f / scale;
    float q = roundf(input[idx] * inv_scale);
    q = fminf(127.0f, fmaxf(-127.0f, q));
    output[idx] = (int8_t)q;
}

extern "C" __global__
void dequantize_rowwise_i8_f32(
    const int8_t* __restrict__ input,  // [rows, D]
    const float* __restrict__ scales,  // [rows]
    float* __restrict__ output,        // [rows, D]
    int rows,
    int d
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = rows * d;
    if (idx >= total) return;
    int row = idx / d;
    output[idx] = ((float)input[idx]) * scales[row];
}
