#pragma once

#include <vector>

namespace mlp_native {

struct NativeLinRegMetrics {
    std::vector<double> loss_per_epoch;
    double learned_w = 0.0;
    double learned_b = 0.0;
    float training_time_ms = 0.0f;
};

struct NativeMlpMetrics {
    std::vector<float> hidden_pre_relu;
    std::vector<float> probs;
    float inference_time_ms = 0.0f;
};

NativeLinRegMetrics run_native_linear_regression_training(
    const std::vector<float>& h_x,
    const std::vector<float>& h_y,
    float init_w,
    float init_b,
    int epochs,
    float learning_rate);

NativeMlpMetrics run_native_mlp_forward(
    const std::vector<float>& h_input,
    const std::vector<float>& h_w1,
    const std::vector<float>& h_w2,
    unsigned int in_dim,
    unsigned int hidden_dim,
    unsigned int out_dim);

}  // namespace mlp_native
