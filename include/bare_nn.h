#pragma once

#include <cuda.h>

namespace bare_nn {

struct MatmulConfig {
    unsigned int BT_M = 64;
    unsigned int BT_N = 64;
    unsigned int BT_K = 16;
    unsigned int WT_X = 8;
    unsigned int WT_Y = 4;
    unsigned int TT_X = 4;
    unsigned int TT_Y = 4;
};

struct Complex {
    float real;
    float imag;
};

// activation_fn.cu
void relu(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n, int block_size = 256, CUstream stream = nullptr);
void sigmoid(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n, int block_size = 256, CUstream stream = nullptr);
void tanh_activation(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n, int block_size = 256, CUstream stream = nullptr);
void softmax(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n, int block_size = 256, CUstream stream = nullptr);
void log_softmax(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n, int block_size = 256, CUstream stream = nullptr);
void leaky_relu(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n, float alpha = 0.01f);
void gelu(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n);

// forward_pass.cu
void vector_add(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_out, unsigned int n, int block_size = 256, CUstream stream = nullptr);
void vector_sub(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_out, unsigned int n, int block_size = 256, CUstream stream = nullptr);
void vector_mul(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_out, unsigned int n, int block_size = 256, CUstream stream = nullptr);
void vector_div(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_out, unsigned int n, int block_size = 256, CUstream stream = nullptr);
void vector_scalar_add(CUdeviceptr d_a, float scalar, CUdeviceptr d_out, unsigned int n, int block_size = 256, CUstream stream = nullptr);
void vector_scalar_mul(CUdeviceptr d_a, float scalar, CUdeviceptr d_out, unsigned int n, int block_size = 256, CUstream stream = nullptr);

void reduce_sum(CUdeviceptr d_input, CUdeviceptr d_out_scalar, unsigned int n, CUstream stream = nullptr);
void reduce_mean(CUdeviceptr d_input, CUdeviceptr d_out_scalar, unsigned int n, CUstream stream = nullptr);
void reduce_max(CUdeviceptr d_input, CUdeviceptr d_out_scalar, unsigned int n, CUstream stream = nullptr);
void reduce_min(CUdeviceptr d_input, CUdeviceptr d_out_scalar, unsigned int n, CUstream stream = nullptr);

void matmul(
    CUdeviceptr d_c,
    CUdeviceptr d_a,
    CUdeviceptr d_b,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    const MatmulConfig& config = MatmulConfig{},
    CUstream stream = nullptr);

void matmul_complex(
    CUdeviceptr d_c,
    CUdeviceptr d_a,
    CUdeviceptr d_b,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    int block_size = 256,
    CUstream stream = nullptr);

void fully_connected_forward(
    CUdeviceptr d_input,
    CUdeviceptr d_weight,
    CUdeviceptr d_output,
    unsigned int in_size,
    unsigned int out_size,
    const MatmulConfig& config = MatmulConfig{},
    CUstream stream = nullptr);

void conv2d_forward();

// losses.cu
void l2_squared_diff(
    CUdeviceptr d_predicted,
    CUdeviceptr d_target,
    CUdeviceptr d_squared_diff,
    unsigned int n,
    int block_size = 256,
    CUstream stream = nullptr);

void l2_norm_reduce(
    CUdeviceptr d_predicted,
    CUdeviceptr d_target,
    CUdeviceptr d_partial_sums,
    unsigned int n,
    int block_size = 256,
    CUstream stream = nullptr);

void mse_reduce(
    CUdeviceptr d_predicted,
    CUdeviceptr d_target,
    CUdeviceptr d_partial_sums,
    unsigned int n,
    int block_size = 256,
    CUstream stream = nullptr);

// backward_pass.cu
void mse_gradient(
    CUdeviceptr d_predicted,
    CUdeviceptr d_target,
    CUdeviceptr d_gradient,
    unsigned int n,
    int block_size = 256,
    CUstream stream = nullptr);

void l2_gradient(
    CUdeviceptr d_predicted,
    CUdeviceptr d_target,
    CUdeviceptr d_gradient,
    unsigned int n,
    int block_size = 256,
    CUstream stream = nullptr);

void relu_grad(
    CUdeviceptr d_input,
    CUdeviceptr d_grad_in,
    CUdeviceptr d_grad_out,
    unsigned int n,
    int block_size = 256,
    CUstream stream = nullptr);

void sigmoid_grad(
    CUdeviceptr d_sigmoid_out,
    CUdeviceptr d_grad_in,
    CUdeviceptr d_grad_out,
    unsigned int n,
    int block_size = 256,
    CUstream stream = nullptr);

void sigmoid_grad_from_input(
    CUdeviceptr d_input,
    CUdeviceptr d_grad_in,
    CUdeviceptr d_grad_out,
    unsigned int n,
    int block_size = 256,
    CUstream stream = nullptr);

void tanh_grad(
    CUdeviceptr d_tanh_out,
    CUdeviceptr d_grad_in,
    CUdeviceptr d_grad_out,
    unsigned int n,
    int block_size = 256,
    CUstream stream = nullptr);

void tanh_grad_from_input(
    CUdeviceptr d_input,
    CUdeviceptr d_grad_in,
    CUdeviceptr d_grad_out,
    unsigned int n,
    int block_size = 256,
    CUstream stream = nullptr);

void leaky_relu_grad(
    CUdeviceptr d_input,
    CUdeviceptr d_grad_in,
    CUdeviceptr d_grad_out,
    unsigned int n,
    float alpha = 0.01f,
    int block_size = 256,
    CUstream stream = nullptr);

void gelu_grad(
    CUdeviceptr d_input,
    CUdeviceptr d_grad_in,
    CUdeviceptr d_grad_out,
    unsigned int n,
    int block_size = 256,
    CUstream stream = nullptr);

void relu_grad_vec4(
    CUdeviceptr d_input,
    CUdeviceptr d_grad_in,
    CUdeviceptr d_grad_out,
    unsigned int n,
    int block_size = 256,
    CUstream stream = nullptr);

void sigmoid_grad_vec4(
    CUdeviceptr d_sigmoid_out,
    CUdeviceptr d_grad_in,
    CUdeviceptr d_grad_out,
    unsigned int n,
    int block_size = 256,
    CUstream stream = nullptr);

void softmax_grad(CUdeviceptr d_softmax_out, CUdeviceptr d_grad_in, CUdeviceptr d_grad_out, unsigned int n);
void fully_connected_backward();

}  // namespace bare_nn
