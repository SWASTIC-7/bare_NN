#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BARE_NN_ENABLE_MLP_CHARTS
#define BARE_NN_ENABLE_MLP_CHARTS 1
#endif

#include <cmath>
#if BARE_NN_ENABLE_MLP_CHARTS
#include <filesystem>
#endif
#include <numeric>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "bare_nn.h"
#if BARE_NN_ENABLE_MLP_CHARTS
#include "charts_api.h"
#endif
#include "cuda_utils.cuh"
#include "native_kernels.cuh"

namespace {

struct LinearRegressionMetrics {
    std::vector<double> loss_per_epoch;
    double learned_w = 0.0;
    double learned_b = 0.0;
    double target_w = 0.0;
    double target_b = 0.0;
    float training_time_ms = 0.0f;
};

struct MlpForwardMetrics {
    std::vector<float> hidden_pre_relu;
    std::vector<float> probs;
    float inference_time_ms = 0.0f;
};

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

LinearRegressionMetrics run_wrapper_linear_regression_training(
    const std::vector<float>& h_x,
    const std::vector<float>& h_y,
    float init_w,
    float init_b,
    int epochs,
    float learning_rate,
    float target_w,
    float target_b) {
    const int kSamples = static_cast<int>(h_x.size());
    std::vector<float> h_grad(kSamples, 0.0f);
    std::vector<float> h_tmp_mul(kSamples, 0.0f);
    std::vector<float> h_sq(kSamples, 0.0f);
    if (kSamples == 0 || h_y.size() != h_x.size() || epochs <= 0) {
        return {};
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

    float w = init_w;
    float b = init_b;
    float last_mse = 0.0f;
    LinearRegressionMetrics metrics;
    metrics.target_w = target_w;
    metrics.target_b = target_b;
    metrics.loss_per_epoch.reserve(static_cast<size_t>(epochs));

    cudaEvent_t start_evt;
    cudaEvent_t stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
    CUDA_CHECK(cudaEventRecord(start_evt));

    for (int epoch = 1; epoch <= epochs; ++epoch) {
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
        metrics.loss_per_epoch.push_back(static_cast<double>(last_mse));

        w -= learning_rate * h_dw;
        b -= learning_rate * h_db;

        if (epoch == 1 || epoch % 100 == 0 || epoch == epochs) {
            printf("[linreg] epoch=%d mse=%.6f w=%.6f b=%.6f\n", epoch, last_mse, w, b);
        }
    }

    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));
    CUDA_CHECK(cudaEventElapsedTime(&metrics.training_time_ms, start_evt, stop_evt));

    printf("[linreg] learned: w=%.6f b=%.6f\n", w, b);
    printf("[linreg] target : w=%.6f b=%.6f\n", target_w, target_b);

    metrics.learned_w = w;
    metrics.learned_b = b;

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);

    cuMemFree(d_x);
    cuMemFree(d_y);
    cuMemFree(d_pred);
    cuMemFree(d_grad);
    cuMemFree(d_sq);
    cuMemFree(d_tmp_mul);

    return metrics;
}

MlpForwardMetrics run_wrapper_mlp_forward(
    const std::vector<float>& h_input,
    const std::vector<float>& h_w1,
    const std::vector<float>& h_w2,
    unsigned int kInDim,
    unsigned int kHiddenDim,
    unsigned int kOutDim) {

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

    cudaEvent_t start_evt;
    cudaEvent_t stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
    CUDA_CHECK(cudaEventRecord(start_evt));

    bare_nn::fully_connected_forward(d_input, d_w1, d_hidden, kInDim, kHiddenDim);
    bare_nn::relu(d_hidden, d_hidden_act, kHiddenDim);
    bare_nn::fully_connected_forward(d_hidden_act, d_w2, d_logits, kHiddenDim, kOutDim);
    bare_nn::softmax(d_logits, d_probs, kOutDim);

    CU_CHECK(cuCtxSynchronize());

    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));

    CU_CHECK(cuMemcpyDtoH(h_hidden.data(), d_hidden, kHiddenDim * sizeof(float)));
    CU_CHECK(cuMemcpyDtoH(h_hidden_act.data(), d_hidden_act, kHiddenDim * sizeof(float)));
    CU_CHECK(cuMemcpyDtoH(h_logits.data(), d_logits, kOutDim * sizeof(float)));
    CU_CHECK(cuMemcpyDtoH(h_probs.data(), d_probs, kOutDim * sizeof(float)));

    print_vector("input", h_input);
    print_vector("hidden_pre_relu", h_hidden);
    print_vector("hidden_post_relu", h_hidden_act);
    print_vector("logits", h_logits);
    print_vector("softmax", h_probs);

    MlpForwardMetrics metrics;
    metrics.hidden_pre_relu = h_hidden;
    metrics.probs = h_probs;
    CUDA_CHECK(cudaEventElapsedTime(&metrics.inference_time_ms, start_evt, stop_evt));

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    cuMemFree(d_input);
    cuMemFree(d_w1);
    cuMemFree(d_hidden);
    cuMemFree(d_hidden_act);
    cuMemFree(d_w2);
    cuMemFree(d_logits);
    cuMemFree(d_probs);

    return metrics;
}

#if BARE_NN_ENABLE_MLP_CHARTS
std::vector<std::string> epoch_labels(size_t n) {
    std::vector<std::string> labels;
    labels.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        labels.push_back(std::to_string(i + 1));
    }
    return labels;
}

void generate_comparison_charts(
    const LinearRegressionMetrics& wrapper_linreg,
    const mlp_native::NativeLinRegMetrics& native_linreg,
    const MlpForwardMetrics& wrapper_mlp,
    const mlp_native::NativeMlpMetrics& native_mlp) {
    std::filesystem::create_directories("examples/MLP/charts");

    bare_nn::charts::ChartConfig cfg;
    cfg.width = 980;
    cfg.height = 620;

    const std::vector<std::string> labels = epoch_labels(wrapper_linreg.loss_per_epoch.size());
    cfg.title = "Training Loss Comparison (Per Epoch)";
    if (!bare_nn::charts::create_multi_line_chart(
            "examples/MLP/charts/training_loss_comparison.svg",
            labels,
            {wrapper_linreg.loss_per_epoch, native_linreg.loss_per_epoch},
            {"Wrapper API", "Native CUDA"},
            cfg)) {
        printf("[charts] failed: training_loss_comparison.svg\n");
    }

    cfg.title = "Training vs Inference Time (ms)";
    if (!bare_nn::charts::create_grouped_bar_chart(
            "examples/MLP/charts/runtime_comparison.svg",
            {"Training", "Inference"},
            {wrapper_linreg.training_time_ms, wrapper_mlp.inference_time_ms},
            {native_linreg.training_time_ms, native_mlp.inference_time_ms},
            "Wrapper API",
            "Native Kernel",
            cfg)) {
        printf("[charts] failed: runtime_comparison.svg\n");
    }

    cfg.title = "Final Parameter Error (abs)";
    if (!bare_nn::charts::create_grouped_bar_chart(
            "examples/MLP/charts/final_param_error.svg",
            {"|w-w*|", "|b-b*|"},
            {
                std::fabs(wrapper_linreg.learned_w - wrapper_linreg.target_w),
                std::fabs(wrapper_linreg.learned_b - wrapper_linreg.target_b)
            },
            {
                std::fabs(native_linreg.learned_w - wrapper_linreg.target_w),
                std::fabs(native_linreg.learned_b - wrapper_linreg.target_b)
            },
            "Wrapper API",
            "Native Kernel",
            cfg)) {
        printf("[charts] failed: final_param_error.svg\n");
    }

    std::vector<double> wrapper_probs;
    std::vector<double> native_probs;
    wrapper_probs.reserve(wrapper_mlp.probs.size());
    native_probs.reserve(native_mlp.probs.size());
    for (float p : wrapper_mlp.probs) wrapper_probs.push_back(static_cast<double>(p) * 100.0);
    for (float p : native_mlp.probs) native_probs.push_back(static_cast<double>(p) * 100.0);

    cfg.title = "Softmax Output Comparison (%)";
    if (!bare_nn::charts::create_grouped_bar_chart(
            "examples/MLP/charts/softmax_comparison.svg",
            {"Class0", "Class1", "Class2"},
            wrapper_probs,
            native_probs,
            "Wrapper API",
            "Native Kernel",
            cfg)) {
        printf("[charts] failed: softmax_comparison.svg\n");
    }

    printf("[charts] wrote SVGs to examples/MLP/charts/\n");
}
#endif

}  // namespace

int main() {
    constexpr int kSamples = 256;
    constexpr int kEpochs = 400;
    constexpr float kLearningRate = 0.08f;
    constexpr float kTargetW = 2.5f;
    constexpr float kTargetB = 1.2f;
    constexpr float kInitW = -0.7f;
    constexpr float kInitB = 0.3f;

    constexpr unsigned int kInDim = 4;
    constexpr unsigned int kHiddenDim = 8;
    constexpr unsigned int kOutDim = 3;

    CUdevice dev;
    CUcontext ctx;
    init_driver_context(&dev, &ctx);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> x_dist(-2.0f, 2.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.05f);
    std::vector<float> h_x(kSamples);
    std::vector<float> h_y(kSamples);
    for (int i = 0; i < kSamples; ++i) {
        h_x[i] = x_dist(rng);
        h_y[i] = kTargetW * h_x[i] + kTargetB + noise_dist(rng);
    }

    std::vector<float> h_input = {0.2f, -0.5f, 0.1f, 0.9f};
    std::vector<float> h_w1 = random_vector(kInDim * kHiddenDim, -0.4f, 0.4f, 7);
    std::vector<float> h_w2 = random_vector(kHiddenDim * kOutDim, -0.3f, 0.3f, 13);

    printf("Training Linear Regression (Wrapper API)....\n");
    LinearRegressionMetrics wrapper_linreg = run_wrapper_linear_regression_training(
        h_x, h_y, kInitW, kInitB, kEpochs, kLearningRate, kTargetW, kTargetB);

    printf("\nTraining Linear Regression (Native CUDA kernels)....\n");
    mlp_native::NativeLinRegMetrics native_linreg = mlp_native::run_native_linear_regression_training(
        h_x, h_y, kInitW, kInitB, kEpochs, kLearningRate);
    printf("[native] learned: w=%.6f b=%.6f\n", native_linreg.learned_w, native_linreg.learned_b);
    printf("[native] target : w=%.6f b=%.6f\n", kTargetW, kTargetB);

    printf("\nInference (Wrapper API)\n");
    MlpForwardMetrics wrapper_mlp = run_wrapper_mlp_forward(h_input, h_w1, h_w2, kInDim, kHiddenDim, kOutDim);

    printf("\nInference (Native CUDA kernels)\n");
    mlp_native::NativeMlpMetrics native_mlp = mlp_native::run_native_mlp_forward(
        h_input, h_w1, h_w2, kInDim, kHiddenDim, kOutDim);

    printf("\n=== Runtime Summary ===\n");
    printf("wrapper training   : %.3f ms\n", wrapper_linreg.training_time_ms);
    printf("native training    : %.3f ms\n", native_linreg.training_time_ms);
    printf("wrapper inference  : %.3f ms\n", wrapper_mlp.inference_time_ms);
    printf("native inference   : %.3f ms\n", native_mlp.inference_time_ms);

    printf("\nCreating charts...\n");
#if BARE_NN_ENABLE_MLP_CHARTS
    generate_comparison_charts(wrapper_linreg, native_linreg, wrapper_mlp, native_mlp);
#else
    (void)wrapper_linreg;
    (void)native_linreg;
    (void)wrapper_mlp;
    (void)native_mlp;
    printf("[charts] disabled (compile with -DBARE_NN_ENABLE_MLP_CHARTS=1 to enable).\n");
#endif

    release_driver_context(dev);
    return 0;
}
