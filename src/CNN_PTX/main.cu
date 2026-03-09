
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

// Error check macro for CUDA Driver API
#define CHECK(call)                                                     \
    do {                                                                \
        CUresult err = call;                                            \
        if (err != CUDA_SUCCESS) {                                      \
            const char* msg;                                            \
            cuGetErrorString(err, &msg);                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                    __FILE__, __LINE__, msg);                           \
            exit(1);                                                    \
        }                                                               \
    } while (0)

// MNIST data
static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// CNN Layers (LeNet-style)
// l_input: 28x28 = 784
// l_c1: Conv 5x5, 6 filters -> 24x24x6 = 3456 output
// l_s1: Pooling 4x4 -> 6x6x6 = 216 output
// l_f: FC 216 -> 10
static Layer l_input(0, 0, 28*28);
static Layer l_c1(5*5, 6, 24*24*6);
static Layer l_s1(4*4, 1, 6*6*6);
static Layer l_f(6*6*6, 10, 10);

// Function declarations
static void learn();
static void test();
static unsigned int classify(double data[28][28]);
static double forward_pass(double data[28][28]);
static double back_pass();

static inline void loaddata()
{
    mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
        &train_set, &train_cnt);
    mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
        &test_set, &test_cnt);
}

int main(int argc, const char **argv)
{
    srand(time(NULL));

    // Initialize CUDA Driver API
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
        return 1;
    }

    // Load PTX modules
    init_activation_ptx("ptx/activation_fn.ptx");
    init_loss_ptx("ptx/losses.ptx");
    init_forward_ptx("ptx/forward_pass.ptx");
    init_backward_ptx("ptx/backward_pass.ptx");

    // Load data and train
    loaddata();
    learn();
    test();

    // Cleanup PTX modules
    cleanup_activation_ptx();
    cleanup_loss_ptx();
    cleanup_forward_ptx();
    cleanup_backward_ptx();

    return 0;
}

// Forward propagation of a single image
static double forward_pass(double data[28][28])
{
    float input[28][28];

    // Convert double to float
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            input[i][j] = (float)data[i][j];
        }
    }

    // Clear layer outputs
    l_input.clear();
    l_c1.clear();
    l_s1.clear();
    l_f.clear();

    clock_t start = clock();

    // Set input
    l_input.setOutput((float*)input);

    // Conv layer C1: 28x28 -> 6x24x24
    // Zero preact first, then accumulate
    launch_memset_zero_ptx(l_c1.preact, l_c1.O);
    launch_conv2d_ptx(l_input.output, l_c1.preact, l_c1.weight, 28, 28, 5, 5, 6);
    launch_add_bias_ptx(l_c1.preact, l_c1.bias, 6, 24*24);
    launch_sigmoid_ptx(l_c1.preact, l_c1.output, l_c1.O);

    // Pooling layer S1: 6x24x24 -> 6x6x6
    launch_memset_zero_ptx(l_s1.preact, l_s1.O);
    launch_pooling_ptx(l_c1.output, l_s1.preact, l_s1.weight, 6, 24, 24, 4, 4);
    launch_add_bias_shared_ptx(l_s1.preact, l_s1.bias, l_s1.O);
    launch_sigmoid_ptx(l_s1.preact, l_s1.output, l_s1.O);

    // FC layer F: 6x6x6=216 -> 10
    launch_memset_zero_ptx(l_f.preact, l_f.O);
    launch_fc_forward_ptx(l_s1.output, l_f.preact, l_f.weight, 6*6*6, 10);
    launch_add_bias_ptx(l_f.preact, l_f.bias, 10, 1);
    launch_sigmoid_ptx(l_f.preact, l_f.output, l_f.O);

    cudaDeviceSynchronize();

    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass()
{
    clock_t start = clock();

    // === FC layer backprop ===
    // Weight gradient: d_weight = d_preact * prev_output
    launch_bp_fc_weight_ptx(l_f.d_weight, l_f.d_preact, l_s1.output, 6*6*6, 10);
    // Bias update
    launch_bp_fc_bias_ptx(l_f.bias, l_f.d_preact, dt, 10);
    // Backprop to S1 output
    launch_bp_fc_output_ptx(l_s1.d_output, l_f.weight, l_f.d_preact, 6*6*6, 10);

    // === Pooling layer backprop ===
    // Sigmoid gradient
    launch_bp_sigmoid_grad_ptx(l_s1.d_preact, l_s1.d_output, l_s1.preact, l_s1.O);
    // Weight gradient
    launch_bp_pooling_weight_ptx(l_s1.d_weight, l_s1.d_preact, l_c1.output, 6, 24, 24, 4, 4);
    // Bias update (shared bias)
    launch_bp_pooling_bias_shared_ptx(l_s1.bias, l_s1.d_preact, dt, l_s1.O);
    // Backprop to C1 output
    launch_bp_pooling_output_ptx(l_c1.d_output, l_s1.weight, l_s1.d_preact, 6, 24, 24, 4, 4);

    // === Conv layer backprop ===
    // Sigmoid gradient
    launch_bp_sigmoid_grad_ptx(l_c1.d_preact, l_c1.d_output, l_c1.preact, l_c1.O);
    // Weight gradient
    launch_bp_conv2d_weight_ptx(l_c1.d_weight, l_c1.d_preact, l_input.output, 28, 28, 5, 5, 6);
    // Bias update
    launch_bp_conv2d_bias_ptx(l_c1.bias, l_c1.d_preact, dt, 6, 24, 24);

    // Apply weight gradients
    launch_apply_grad_ptx(l_f.weight, l_f.d_weight, dt, l_f.M * l_f.N);
    launch_apply_grad_ptx(l_s1.weight, l_s1.d_weight, dt, l_s1.M * l_s1.N);
    launch_apply_grad_ptx(l_c1.weight, l_c1.d_weight, dt, l_c1.M * l_c1.N);

    cudaDeviceSynchronize();

    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

// Training loop
static void learn()
{
    float err;
    int iter = 50;
    double time_taken = 0.0;

    fprintf(stdout, "Learning...\n");

    while (iter-- > 0) {
        err = 0.0f;

        for (unsigned int i = 0; i < train_cnt; ++i) {
            float tmp_err;

            // Forward pass
            time_taken += forward_pass(train_set[i].data);

            // Clear gradients
            l_f.bp_clear();
            l_s1.bp_clear();
            l_c1.bp_clear();

            // Compute error: d_preact = (label == i ? 1.0 : 0.0) - output
            launch_make_error_ptx(l_f.d_preact, l_f.output, train_set[i].label, 10);

            // Compute L2 norm of error (manual reduction on CPU for simplicity)
            float h_d_preact[10];
            cudaMemcpy(h_d_preact, l_f.d_preact, sizeof(float) * 10, cudaMemcpyDeviceToHost);
            tmp_err = 0.0f;
            for (int j = 0; j < 10; ++j) {
                tmp_err += h_d_preact[j] * h_d_preact[j];
            }
            tmp_err = sqrtf(tmp_err);
            err += tmp_err;

            // Backward pass
            time_taken += back_pass();
        }

        err /= train_cnt;
        fprintf(stdout, "Epoch %d: error = %e, time = %lf\n", 50 - iter, err, time_taken);

        if (err < threshold) {
            fprintf(stdout, "Training complete, error below threshold\n\n");
            break;
        }
    }

    fprintf(stdout, "\nTotal training time: %lf seconds\n", time_taken);
}

// Classify a single image
static unsigned int classify(double data[28][28])
{
    float res[10];

    forward_pass(data);

    cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

    unsigned int max_idx = 0;
    for (int i = 1; i < 10; ++i) {
        if (res[i] > res[max_idx]) {
            max_idx = i;
        }
    }

    return max_idx;
}

// Test on test dataset
static void test()
{
    int errors = 0;

    fprintf(stdout, "Testing...\n");

    for (unsigned int i = 0; i < test_cnt; ++i) {
        if (classify(test_set[i].data) != test_set[i].label) {
            ++errors;
        }
    }

    fprintf(stdout, "Test Error Rate: %.2lf%% (%d/%u)\n",
        (double)errors / (double)test_cnt * 100.0, errors, test_cnt);
}