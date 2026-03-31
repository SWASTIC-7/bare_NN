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

namespace {

struct LinearRegressionMetrics {
    std::vector<std::string> epoch_labels;
    std::vector<double> mse_curve;
    std::vector<double> abs_dw_curve;
    double learned_w = 0.0;
    double learned_b = 0.0;
    double target_w = 0.0;
    double target_b = 0.0;
};

struct MlpForwardMetrics {
    std::vector<float> hidden_pre_relu;
    std::vector<float> probs;
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

LinearRegressionMetrics run_linear_regression_training_demo() {
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
    LinearRegressionMetrics metrics;
    metrics.target_w = kTrueW;
    metrics.target_b = kTrueB;

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

        if (epoch == 1 || epoch % 10 == 0 || epoch == kEpochs) {
            metrics.epoch_labels.push_back(std::to_string(epoch));
            metrics.mse_curve.push_back(last_mse);
            metrics.abs_dw_curve.push_back(std::fabs(h_dw));
        }

        w -= kLearningRate * h_dw;
        b -= kLearningRate * h_db;

        if (epoch == 1 || epoch % 100 == 0 || epoch == kEpochs) {
            printf("[linreg] epoch=%d mse=%.6f w=%.6f b=%.6f\n", epoch, last_mse, w, b);
        }
    }

    printf("[linreg] learned: w=%.6f b=%.6f\n", w, b);
    printf("[linreg] target : w=%.6f b=%.6f\n", kTrueW, kTrueB);

    metrics.learned_w = w;
    metrics.learned_b = b;

    cuMemFree(d_x);
    cuMemFree(d_y);
    cuMemFree(d_pred);
    cuMemFree(d_grad);
    cuMemFree(d_sq);
    cuMemFree(d_tmp_mul);

    return metrics;
}

MlpForwardMetrics run_mlp_forward_demo() {
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

    MlpForwardMetrics metrics;
    metrics.hidden_pre_relu = h_hidden;
    metrics.probs = h_probs;

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
void generate_mlp_charts(const LinearRegressionMetrics& linreg, const MlpForwardMetrics& mlp) {
    std::filesystem::create_directories("examples/MLP/charts");

    bare_nn::charts::ChartConfig cfg;
    cfg.width = 980;
    cfg.height = 620;

    cfg.title = "Linear Regression: MSE During Training";
    if (!bare_nn::charts::create_line_chart(
            "examples/MLP/charts/linreg_mse_line.svg", linreg.epoch_labels, linreg.mse_curve, cfg)) {
        printf("[charts] failed: linreg_mse_line.svg\n");
    }

    cfg.title = "Linear Regression: |dw| by Checkpoint";
    if (!bare_nn::charts::create_bar_chart(
            "examples/MLP/charts/linreg_grad_bar.svg", linreg.epoch_labels, linreg.abs_dw_curve, cfg)) {
        printf("[charts] failed: linreg_grad_bar.svg\n");
    }

    std::vector<std::string> class_labels = {"Class 0", "Class 1", "Class 2"};
    std::vector<double> probs_pct;
    probs_pct.reserve(mlp.probs.size());
    for (float p : mlp.probs) {
        probs_pct.push_back(static_cast<double>(p) * 100.0);
    }

    cfg.title = "MLP Output Distribution (Softmax %)";
    if (!bare_nn::charts::create_pie_chart(
            "examples/MLP/charts/mlp_softmax_pie.svg", class_labels, probs_pct, cfg)) {
        printf("[charts] failed: mlp_softmax_pie.svg\n");
    }

    std::vector<std::string> hidden_labels;
    std::vector<double> hidden_pos;
    std::vector<double> hidden_neg;
    hidden_labels.reserve(mlp.hidden_pre_relu.size());
    hidden_pos.reserve(mlp.hidden_pre_relu.size());
    hidden_neg.reserve(mlp.hidden_pre_relu.size());
    for (size_t i = 0; i < mlp.hidden_pre_relu.size(); ++i) {
        hidden_labels.push_back("H" + std::to_string(i));
        double v = static_cast<double>(mlp.hidden_pre_relu[i]);
        hidden_pos.push_back(v > 0.0 ? v : 0.0);
        hidden_neg.push_back(v < 0.0 ? -v : 0.0);
    }

    cfg.title = "Hidden Pre-ReLU Magnitude Split";
    if (!bare_nn::charts::create_stacked_bar_chart(
            "examples/MLP/charts/mlp_hidden_stacked.svg",
            hidden_labels,
            {"positive", "negative(abs)"},
            {hidden_pos, hidden_neg},
            cfg)) {
        printf("[charts] failed: mlp_hidden_stacked.svg\n");
    }


    printf("[charts] wrote SVGs to examples/MLP/charts/\n");
}
#endif

}  // namespace

int main() {
    CUdevice dev;
    CUcontext ctx;
    init_driver_context(&dev, &ctx);

    printf("Training Linear Regression....\n");
    LinearRegressionMetrics linreg_metrics = run_linear_regression_training_demo();

    printf("\nInference\n");
    MlpForwardMetrics mlp_metrics = run_mlp_forward_demo();

    printf("\nCreating charts...\n");
#if BARE_NN_ENABLE_MLP_CHARTS
    generate_mlp_charts(linreg_metrics, mlp_metrics);
#else
    (void)linreg_metrics;
    (void)mlp_metrics;
    printf("[charts] disabled (compile with -DBARE_NN_ENABLE_MLP_CHARTS=1 to enable).\n");
#endif

    release_driver_context(dev);
    return 0;
}
