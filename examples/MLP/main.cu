#include <cuda.h>
#include <cuda_runtime.h>

#include <numeric>
#include <cstdio>
#include <random>
#include <vector>

#include "bare_nn.h"
#include "cuda_utils.cuh"

namespace {

void init_driver_context(CUdevice* dev, CUcontext* ctx) {
    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(dev, 0));
    CU_CHECK(cuDevicePrimaryCtxRetain(ctx, *dev));
    CU_CHECK(cuCtxSetCurrent(*ctx));
}

void release_driver_context(CUdevice dev) {
    CU_CHECK(cuDevicePrimaryCtxRelease(dev));
}

std::vector<float> random_vector(size_t n, float lo, float hi, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);

    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = dist(rng);
    }
    return out;
}

void print_vector(const char* name, const std::vector<float>& v) {
    printf("%s: [", name);
    for (size_t i = 0; i < v.size(); ++i) {
        printf("%.6f", v[i]);
        if (i + 1 != v.size()) {
            printf(", ");
        }
    }
    printf("]\n");
}

void run_linear_regression_training_demo() {
    constexpr int kSamples = 256;
    constexpr float kTrueW = 2.5f;
    constexpr float kTrueB = 1.2f;
    constexpr int kEpochs = 400;
    constexpr float kLearningRate = 0.08f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> x_dist(-2.0f, 2.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.05f);

    std::vector<float> h_x(kSamples);
    std::vector<float> h_y(kSamples);
    std::vector<float> h_grad(kSamples, 0.0f);
    std::vector<float> h_tmp_mul(kSamples, 0.0f);
    std::vector<float> h_sq(kSamples, 0.0f);
    for (int i = 0; i < kSamples; ++i) {
        h_x[i] = x_dist(rng);
        h_y[i] = kTrueW * h_x[i] + kTrueB + noise_dist(rng);
    }

    CUdeviceptr d_x;
    CUdeviceptr d_y;
    CUdeviceptr d_pred;
    CUdeviceptr d_grad;
    CUdeviceptr d_sq;
    CUdeviceptr d_tmp_mul;
    CU_CHECK(cuMemAlloc(&d_x, kSamples * sizeof(float)));
    CU_CHECK(cuMemAlloc(&d_y, kSamples * sizeof(float)));
    CU_CHECK(cuMemAlloc(&d_pred, kSamples * sizeof(float)));
    CU_CHECK(cuMemAlloc(&d_grad, kSamples * sizeof(float)));
    CU_CHECK(cuMemAlloc(&d_sq, kSamples * sizeof(float)));
    CU_CHECK(cuMemAlloc(&d_tmp_mul, kSamples * sizeof(float)));

    CU_CHECK(cuMemcpyHtoD(d_x, h_x.data(), kSamples * sizeof(float)));
    CU_CHECK(cuMemcpyHtoD(d_y, h_y.data(), kSamples * sizeof(float)));

    float w = -0.7f;
    float b = 0.3f;
    float last_mse = 0.0f;

    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        // pred = w * x + b
        bare_nn::vector_scalar_mul(d_x, w, d_pred, kSamples);
        bare_nn::vector_scalar_add(d_pred, b, d_pred, kSamples);

        // grad = d/dpred MSE(pred, y) = 2 * (pred - y) / n
        bare_nn::mse_gradient(d_pred, d_y, d_grad, kSamples);

        // dw = sum(grad * x)
        bare_nn::vector_mul(d_grad, d_x, d_tmp_mul, kSamples);

        // mse = mean((pred - y)^2)
        bare_nn::l2_squared_diff(d_pred, d_y, d_sq, kSamples);

        CU_CHECK(cuCtxSynchronize());

        CU_CHECK(cuMemcpyDtoH(h_grad.data(), d_grad, kSamples * sizeof(float)));
        CU_CHECK(cuMemcpyDtoH(h_tmp_mul.data(), d_tmp_mul, kSamples * sizeof(float)));
        CU_CHECK(cuMemcpyDtoH(h_sq.data(), d_sq, kSamples * sizeof(float)));

        float h_dw = std::accumulate(h_tmp_mul.begin(), h_tmp_mul.end(), 0.0f);
        float h_db = std::accumulate(h_grad.begin(), h_grad.end(), 0.0f);
        float h_sq_sum = std::accumulate(h_sq.begin(), h_sq.end(), 0.0f);
        last_mse = h_sq_sum / static_cast<float>(kSamples);

        w -= kLearningRate * h_dw;
        b -= kLearningRate * h_db;

        if (epoch == 1 || epoch % 100 == 0 || epoch == kEpochs) {
            printf("[linreg] epoch=%d mse=%.6f w=%.6f b=%.6f\n", epoch, last_mse, w, b);
        }
    }

    printf("[linreg] learned: w=%.6f b=%.6f\n", w, b);
    printf("[linreg] target : w=%.6f b=%.6f\n", kTrueW, kTrueB);

    cuMemFree(d_x);
    cuMemFree(d_y);
    cuMemFree(d_pred);
    cuMemFree(d_grad);
    cuMemFree(d_sq);
    cuMemFree(d_tmp_mul);
}

void run_mlp_forward_demo() {
    constexpr unsigned int kInDim = 4;
    constexpr unsigned int kHiddenDim = 8;
    constexpr unsigned int kOutDim = 3;

    std::vector<float> h_input = {0.2f, -0.5f, 0.1f, 0.9f};
    std::vector<float> h_w1 = random_vector(kInDim * kHiddenDim, -0.4f, 0.4f, 7);
    std::vector<float> h_w2 = random_vector(kHiddenDim * kOutDim, -0.3f, 0.3f, 13);

    std::vector<float> h_hidden(kHiddenDim, 0.0f);
    std::vector<float> h_hidden_act(kHiddenDim, 0.0f);
    std::vector<float> h_logits(kOutDim, 0.0f);
    std::vector<float> h_probs(kOutDim, 0.0f);

    CUdeviceptr d_input;
    CUdeviceptr d_w1;
    CUdeviceptr d_hidden;
    CUdeviceptr d_hidden_act;
    CUdeviceptr d_w2;
    CUdeviceptr d_logits;
    CUdeviceptr d_probs;

    CU_CHECK(cuMemAlloc(&d_input, kInDim * sizeof(float)));
    CU_CHECK(cuMemAlloc(&d_w1, kInDim * kHiddenDim * sizeof(float)));
    CU_CHECK(cuMemAlloc(&d_hidden, kHiddenDim * sizeof(float)));
    CU_CHECK(cuMemAlloc(&d_hidden_act, kHiddenDim * sizeof(float)));
    CU_CHECK(cuMemAlloc(&d_w2, kHiddenDim * kOutDim * sizeof(float)));
    CU_CHECK(cuMemAlloc(&d_logits, kOutDim * sizeof(float)));
    CU_CHECK(cuMemAlloc(&d_probs, kOutDim * sizeof(float)));

    CU_CHECK(cuMemcpyHtoD(d_input, h_input.data(), kInDim * sizeof(float)));
    CU_CHECK(cuMemcpyHtoD(d_w1, h_w1.data(), kInDim * kHiddenDim * sizeof(float)));
    CU_CHECK(cuMemcpyHtoD(d_w2, h_w2.data(), kHiddenDim * kOutDim * sizeof(float)));

    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_hidden), 0, kHiddenDim * sizeof(float)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_hidden_act), 0, kHiddenDim * sizeof(float)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_logits), 0, kOutDim * sizeof(float)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_probs), 0, kOutDim * sizeof(float)));

    bare_nn::fully_connected_forward(d_input, d_w1, d_hidden, kInDim, kHiddenDim);
    bare_nn::relu(d_hidden, d_hidden_act, kHiddenDim);
    bare_nn::fully_connected_forward(d_hidden_act, d_w2, d_logits, kHiddenDim, kOutDim);
    bare_nn::softmax(d_logits, d_probs, kOutDim);

    CU_CHECK(cuCtxSynchronize());

    CU_CHECK(cuMemcpyDtoH(h_hidden.data(), d_hidden, kHiddenDim * sizeof(float)));
    CU_CHECK(cuMemcpyDtoH(h_hidden_act.data(), d_hidden_act, kHiddenDim * sizeof(float)));
    CU_CHECK(cuMemcpyDtoH(h_logits.data(), d_logits, kOutDim * sizeof(float)));
    CU_CHECK(cuMemcpyDtoH(h_probs.data(), d_probs, kOutDim * sizeof(float)));

    print_vector("input", h_input);
    print_vector("hidden_pre_relu", h_hidden);
    print_vector("hidden_post_relu", h_hidden_act);
    print_vector("logits", h_logits);
    print_vector("softmax", h_probs);

    cuMemFree(d_input);
    cuMemFree(d_w1);
    cuMemFree(d_hidden);
    cuMemFree(d_hidden_act);
    cuMemFree(d_w2);
    cuMemFree(d_logits);
    cuMemFree(d_probs);
}

}  // namespace

int main() {
    CUdevice dev;
    CUcontext ctx;
    init_driver_context(&dev, &ctx);

    printf("=== Linear Regression Training (GPU) ===\n");
    run_linear_regression_training_demo();

    printf("\n=== MLP Forward Demo (PTX wrappers) ===\n");
    run_mlp_forward_demo();

    release_driver_context(dev);
    return 0;
}
