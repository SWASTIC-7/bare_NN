#include <cuda.h>
#include <cuda_runtime.h>

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

}  // namespace

int main() {
    constexpr unsigned int kInDim = 4;
    constexpr unsigned int kHiddenDim = 8;
    constexpr unsigned int kOutDim = 3;

    CUdevice dev;
    CUcontext ctx;
    init_driver_context(&dev, &ctx);

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

    release_driver_context(dev);
    return 0;
}
