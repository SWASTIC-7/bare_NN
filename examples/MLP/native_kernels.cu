#include "native_kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "cuda_utils.cuh"

namespace mlp_native {
namespace {

__global__ void k_vec_scalar_mul(const float* a, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * s;
    }
}

__global__ void k_vec_scalar_add(const float* a, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + s;
    }
}

__global__ void k_mse_grad(const float* pred, const float* target, float* grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad[i] = 2.0f * (pred[i] - target[i]) / static_cast<float>(n);
    }
}

__global__ void k_vec_mul(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}

__global__ void k_sq_diff(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d = a[i] - b[i];
        out[i] = d * d;
    }
}

__global__ void k_fc_forward_1xN(
    const float* input,
    const float* weight,
    float* output,
    unsigned int in_size,
    unsigned int out_size) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_size) {
        return;
    }
    float acc = 0.0f;
    for (unsigned int i = 0; i < in_size; ++i) {
        acc += input[i] * weight[i * out_size + j];
    }
    output[j] = acc;
}

__global__ void k_relu(const float* in, float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

__global__ void k_softmax_single(const float* in, float* out, unsigned int n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    float max_v = in[0];
    for (unsigned int i = 1; i < n; ++i) {
        if (in[i] > max_v) {
            max_v = in[i];
        }
    }

    float sum = 0.0f;
    for (unsigned int i = 0; i < n; ++i) {
        out[i] = expf(in[i] - max_v);
        sum += out[i];
    }

    for (unsigned int i = 0; i < n; ++i) {
        out[i] /= sum;
    }
}

}  // namespace

NativeLinRegMetrics run_native_linear_regression_training(
    const std::vector<float>& h_x,
    const std::vector<float>& h_y,
    float init_w,
    float init_b,
    int epochs,
    float learning_rate) {
    NativeLinRegMetrics metrics;
    if (h_x.empty() || h_x.size() != h_y.size() || epochs <= 0) {
        return metrics;
    }

    int n = static_cast<int>(h_x.size());
    std::vector<float> h_grad(static_cast<size_t>(n), 0.0f);
    std::vector<float> h_tmp(static_cast<size_t>(n), 0.0f);
    std::vector<float> h_sq(static_cast<size_t>(n), 0.0f);

    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_pred = nullptr;
    float* d_grad = nullptr;
    float* d_tmp = nullptr;
    float* d_sq = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pred, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sq, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    float w = init_w;
    float b = init_b;

    const int block = 256;
    const int grid = (n + block - 1) / block;

    cudaEvent_t start_evt;
    cudaEvent_t stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
    CUDA_CHECK(cudaEventRecord(start_evt));

    metrics.loss_per_epoch.reserve(static_cast<size_t>(epochs));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        k_vec_scalar_mul<<<grid, block>>>(d_x, w, d_pred, n);
        k_vec_scalar_add<<<grid, block>>>(d_pred, b, d_pred, n);
        k_mse_grad<<<grid, block>>>(d_pred, d_y, d_grad, n);
        k_vec_mul<<<grid, block>>>(d_grad, d_x, d_tmp, n);
        k_sq_diff<<<grid, block>>>(d_pred, d_y, d_sq, n);
        KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_grad.data(), d_grad, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tmp.data(), d_tmp, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sq.data(), d_sq, n * sizeof(float), cudaMemcpyDeviceToHost));

        float dw = std::accumulate(h_tmp.begin(), h_tmp.end(), 0.0f);
        float db = std::accumulate(h_grad.begin(), h_grad.end(), 0.0f);
        float mse = std::accumulate(h_sq.begin(), h_sq.end(), 0.0f) / static_cast<float>(n);

        metrics.loss_per_epoch.push_back(static_cast<double>(mse));
        w -= learning_rate * dw;
        b -= learning_rate * db;
    }

    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));
    CUDA_CHECK(cudaEventElapsedTime(&metrics.training_time_ms, start_evt, stop_evt));

    metrics.learned_w = w;
    metrics.learned_b = b;

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_pred);
    cudaFree(d_grad);
    cudaFree(d_tmp);
    cudaFree(d_sq);

    return metrics;
}

NativeMlpMetrics run_native_mlp_forward(
    const std::vector<float>& h_input,
    const std::vector<float>& h_w1,
    const std::vector<float>& h_w2,
    unsigned int in_dim,
    unsigned int hidden_dim,
    unsigned int out_dim) {
    NativeMlpMetrics metrics;
    if (h_input.size() != in_dim || h_w1.size() != in_dim * hidden_dim || h_w2.size() != hidden_dim * out_dim) {
        return metrics;
    }

    std::vector<float> h_hidden(hidden_dim, 0.0f);
    std::vector<float> h_probs(out_dim, 0.0f);

    float* d_input = nullptr;
    float* d_w1 = nullptr;
    float* d_hidden = nullptr;
    float* d_hidden_act = nullptr;
    float* d_w2 = nullptr;
    float* d_logits = nullptr;
    float* d_probs = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, in_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w1, in_dim * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden_act, hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, hidden_dim * out_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, out_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probs, out_dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), in_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(), in_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(), hidden_dim * out_dim * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start_evt;
    cudaEvent_t stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
    CUDA_CHECK(cudaEventRecord(start_evt));

    int block = 256;
    int hidden_grid = (static_cast<int>(hidden_dim) + block - 1) / block;
    int out_grid = (static_cast<int>(out_dim) + block - 1) / block;

    k_fc_forward_1xN<<<hidden_grid, block>>>(d_input, d_w1, d_hidden, in_dim, hidden_dim);
    k_relu<<<hidden_grid, block>>>(d_hidden, d_hidden_act, hidden_dim);
    k_fc_forward_1xN<<<out_grid, block>>>(d_hidden_act, d_w2, d_logits, hidden_dim, out_dim);
    k_softmax_single<<<1, 1>>>(d_logits, d_probs, out_dim);
    KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));
    CUDA_CHECK(cudaEventElapsedTime(&metrics.inference_time_ms, start_evt, stop_evt));

    CUDA_CHECK(cudaMemcpy(h_hidden.data(), d_hidden, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_probs.data(), d_probs, out_dim * sizeof(float), cudaMemcpyDeviceToHost));

    metrics.hidden_pre_relu = std::move(h_hidden);
    metrics.probs = std::move(h_probs);

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    cudaFree(d_input);
    cudaFree(d_w1);
    cudaFree(d_hidden);
    cudaFree(d_hidden_act);
    cudaFree(d_w2);
    cudaFree(d_logits);
    cudaFree(d_probs);

    return metrics;
}

}  // namespace mlp_native
