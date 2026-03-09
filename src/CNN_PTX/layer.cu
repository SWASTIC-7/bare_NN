#include "layer.h"
#include <cstdio>

// PTX module and kernel handles for activation functions
static CUmodule activation_module = nullptr;
static CUfunction sigmoid_kernel = nullptr;

// PTX module and kernel handles for loss functions
static CUmodule loss_module = nullptr;
static CUfunction make_error_onehot_kernel = nullptr;
static CUfunction mse_gradient_kernel = nullptr;

// PTX module and kernel handles for forward pass
static CUmodule forward_module = nullptr;
static CUfunction conv2d_kernel = nullptr;
static CUfunction conv2d_mc_kernel = nullptr;
static CUfunction add_bias_kernel = nullptr;
static CUfunction add_bias_shared_kernel = nullptr;
static CUfunction pooling_kernel = nullptr;
static CUfunction fc_forward_kernel = nullptr;
static CUfunction apply_grad_kernel = nullptr;
static CUfunction memset_zero_kernel = nullptr;

// PTX module and kernel handles for backward pass
static CUmodule backward_module = nullptr;
static CUfunction bp_sigmoid_grad_kernel = nullptr;
static CUfunction bp_fc_weight_kernel = nullptr;
static CUfunction bp_fc_bias_kernel = nullptr;
static CUfunction bp_fc_output_kernel = nullptr;
static CUfunction bp_pooling_weight_kernel = nullptr;
static CUfunction bp_pooling_bias_shared_kernel = nullptr;
static CUfunction bp_pooling_output_kernel = nullptr;
static CUfunction bp_conv2d_weight_kernel = nullptr;
static CUfunction bp_conv2d_bias_kernel = nullptr;

// Error check macro for CUDA Driver API
#define PTX_CHECK(call)                                                 \
    do {                                                                \
        CUresult err = call;                                            \
        if (err != CUDA_SUCCESS) {                                      \
            const char* msg;                                            \
            cuGetErrorString(err, &msg);                                \
            fprintf(stderr, "CUDA PTX error %s:%d: %s\n",               \
                    __FILE__, __LINE__, msg);                           \
        }                                                               \
    } while (0)


// Constructor
Layer::Layer(int M, int N, int O)
{
	this->M = M;
	this->N = N;
	this->O = O;

	float h_bias[N];
	float h_weight[N][M];

	output = NULL;
	preact = NULL;
	bias   = NULL;
	weight = NULL;

	for (int i = 0; i < N; ++i) {
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
		/*h_bias[i] = 0.0f;*/

		for (int j = 0; j < M; ++j) {
			h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
			/*h_weight[i][j] = 0.05f;*/
		}
	}

	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&preact, sizeof(float) * O);

	cudaMalloc(&bias, sizeof(float) * N);

	cudaMalloc(&weight, sizeof(float) * M * N);

	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_weight, sizeof(float) * M * N);

	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// Destructor
Layer::~Layer()
{
	cudaFree(output);
	cudaFree(preact);

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_weight);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
	cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * O);
	cudaMemset(preact, 0x00, sizeof(float) * O);
}

void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * O);
	cudaMemset(d_preact, 0x00, sizeof(float) * O);
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}

// Initialize PTX activation kernels - call once before using
void init_activation_ptx(const char* ptx_path)
{
    if (activation_module != nullptr) return; // Already initialized
    
    PTX_CHECK(cuModuleLoad(&activation_module, ptx_path));
    PTX_CHECK(cuModuleGetFunction(&sigmoid_kernel, activation_module, "sigmoid"));
}

// Initialize PTX loss kernels - call once before using
void init_loss_ptx(const char* ptx_path)
{
    if (loss_module != nullptr) return; // Already initialized
    
    PTX_CHECK(cuModuleLoad(&loss_module, ptx_path));
    PTX_CHECK(cuModuleGetFunction(&make_error_onehot_kernel, loss_module, "make_error_onehot"));
    PTX_CHECK(cuModuleGetFunction(&mse_gradient_kernel, loss_module, "mse_gradient"));
}

// Cleanup PTX modules - call at end of program
void cleanup_activation_ptx()
{
    if (activation_module != nullptr) {
        cuModuleUnload(activation_module);
        activation_module = nullptr;
        sigmoid_kernel = nullptr;
    }
}

void cleanup_loss_ptx()
{
    if (loss_module != nullptr) {
        cuModuleUnload(loss_module);
        loss_module = nullptr;
        make_error_onehot_kernel = nullptr;
        mse_gradient_kernel = nullptr;
    }
}

// Initialize PTX forward pass kernels - call once before using
void init_forward_ptx(const char* ptx_path)
{
    if (forward_module != nullptr) return; // Already initialized
    
    PTX_CHECK(cuModuleLoad(&forward_module, ptx_path));
    PTX_CHECK(cuModuleGetFunction(&conv2d_kernel, forward_module, "conv2d"));
    PTX_CHECK(cuModuleGetFunction(&conv2d_mc_kernel, forward_module, "conv2d_multi_channel"));
    PTX_CHECK(cuModuleGetFunction(&add_bias_kernel, forward_module, "add_bias"));
    PTX_CHECK(cuModuleGetFunction(&add_bias_shared_kernel, forward_module, "add_bias_shared"));
    PTX_CHECK(cuModuleGetFunction(&pooling_kernel, forward_module, "pooling"));
    PTX_CHECK(cuModuleGetFunction(&fc_forward_kernel, forward_module, "fc_forward"));
    PTX_CHECK(cuModuleGetFunction(&apply_grad_kernel, forward_module, "apply_grad"));
    PTX_CHECK(cuModuleGetFunction(&memset_zero_kernel, forward_module, "memset_zero"));
}

void cleanup_forward_ptx()
{
    if (forward_module != nullptr) {
        cuModuleUnload(forward_module);
        forward_module = nullptr;
        conv2d_kernel = nullptr;
        conv2d_mc_kernel = nullptr;
        add_bias_kernel = nullptr;
        add_bias_shared_kernel = nullptr;
        pooling_kernel = nullptr;
        fc_forward_kernel = nullptr;
        apply_grad_kernel = nullptr;
        memset_zero_kernel = nullptr;
    }
}

// Initialize PTX backward pass kernels - call once before using
void init_backward_ptx(const char* ptx_path)
{
    if (backward_module != nullptr) return; // Already initialized
    
    PTX_CHECK(cuModuleLoad(&backward_module, ptx_path));
    PTX_CHECK(cuModuleGetFunction(&bp_sigmoid_grad_kernel, backward_module, "bp_sigmoid_grad"));
    PTX_CHECK(cuModuleGetFunction(&bp_fc_weight_kernel, backward_module, "bp_fc_weight"));
    PTX_CHECK(cuModuleGetFunction(&bp_fc_bias_kernel, backward_module, "bp_fc_bias"));
    PTX_CHECK(cuModuleGetFunction(&bp_fc_output_kernel, backward_module, "bp_fc_output"));
    PTX_CHECK(cuModuleGetFunction(&bp_pooling_weight_kernel, backward_module, "bp_pooling_weight"));
    PTX_CHECK(cuModuleGetFunction(&bp_pooling_bias_shared_kernel, backward_module, "bp_pooling_bias_shared"));
    PTX_CHECK(cuModuleGetFunction(&bp_pooling_output_kernel, backward_module, "bp_pooling_output"));
    PTX_CHECK(cuModuleGetFunction(&bp_conv2d_weight_kernel, backward_module, "bp_conv2d_weight"));
    PTX_CHECK(cuModuleGetFunction(&bp_conv2d_bias_kernel, backward_module, "bp_conv2d_bias"));
}

void cleanup_backward_ptx()
{
    if (backward_module != nullptr) {
        cuModuleUnload(backward_module);
        backward_module = nullptr;
        bp_sigmoid_grad_kernel = nullptr;
        bp_fc_weight_kernel = nullptr;
        bp_fc_bias_kernel = nullptr;
        bp_fc_output_kernel = nullptr;
        bp_pooling_weight_kernel = nullptr;
        bp_pooling_bias_shared_kernel = nullptr;
        bp_pooling_output_kernel = nullptr;
        bp_conv2d_weight_kernel = nullptr;
        bp_conv2d_bias_kernel = nullptr;
    }
}

// Launch sigmoid kernel from PTX: output[i] = 1 / (1 + exp(-input[i]))
void launch_sigmoid_ptx(float* d_input, float* d_output, int n, int block_size)
{
    if (sigmoid_kernel == nullptr) {
        fprintf(stderr, "Error: PTX activation not initialized. Call init_activation_ptx() first.\n");
        return;
    }
    
    int grid_size = (n + block_size - 1) / block_size;
    unsigned int n_u32 = static_cast<unsigned int>(n);
    
    void* args[] = {
        &d_input,
        &d_output,
        &n_u32
    };
    
    PTX_CHECK(cuLaunchKernel(
        sigmoid_kernel,
        grid_size, 1, 1,    // grid dims
        block_size, 1, 1,   // block dims
        0,                  // shared memory
        0,                  // stream
        args,
        nullptr
    ));
}

// Launch make_error_onehot kernel from PTX: error[i] = (label == i ? 1.0 : 0.0) - output[i]
void launch_make_error_ptx(float* d_error, float* d_output, unsigned int label, int n, int block_size)
{
    if (make_error_onehot_kernel == nullptr) {
        fprintf(stderr, "Error: PTX loss not initialized. Call init_loss_ptx() first.\n");
        return;
    }
    
    int grid_size = (n + block_size - 1) / block_size;
    unsigned int n_u32 = static_cast<unsigned int>(n);
    
    void* args[] = {
        &d_error,
        &d_output,
        &label,
        &n_u32
    };
    
    PTX_CHECK(cuLaunchKernel(
        make_error_onehot_kernel,
        grid_size, 1, 1,    // grid dims
        block_size, 1, 1,   // block dims
        0,                  // shared memory
        0,                  // stream
        args,
        nullptr
    ));
}

// Launch MSE gradient kernel from PTX: gradient[i] = (2/n) * (predicted[i] - target[i])
void launch_mse_gradient_ptx(float* d_predicted, float* d_target, float* d_gradient, int n, int block_size)
{
    if (mse_gradient_kernel == nullptr) {
        fprintf(stderr, "Error: PTX loss not initialized. Call init_loss_ptx() first.\n");
        return;
    }
    
    int grid_size = (n + block_size - 1) / block_size;
    unsigned int n_u32 = static_cast<unsigned int>(n);
    
    void* args[] = {
        &d_predicted,
        &d_target,
        &d_gradient,
        &n_u32
    };
    
    PTX_CHECK(cuLaunchKernel(
        mse_gradient_kernel,
        grid_size, 1, 1,    // grid dims
        block_size, 1, 1,   // block dims
        0,                  // shared memory
        0,                  // stream
        args,
        nullptr
    ));
}

// ============================================================================
// General Forward Pass PTX Launch Functions
// ============================================================================

// conv2d: General 2D convolution for single-channel input
// input[in_h][in_w] * weight[num_filters][kh][kw] -> output[num_filters][out_h][out_w]
void launch_conv2d_ptx(float* d_input, float* d_output, float* d_weight,
                       int in_h, int in_w, int kh, int kw, int num_filters,
                       int block_size)
{
    if (conv2d_kernel == nullptr) {
        fprintf(stderr, "Error: PTX forward not initialized. Call init_forward_ptx() first.\n");
        return;
    }
    
    int out_h = in_h - kh + 1;
    int out_w = in_w - kw + 1;
    unsigned int N = kh * kw * num_filters * out_h * out_w;
    int grid_size = (N + block_size - 1) / block_size;
    
    unsigned int u_in_h = in_h, u_in_w = in_w, u_kh = kh, u_kw = kw;
    unsigned int u_num_filters = num_filters, u_out_h = out_h, u_out_w = out_w;
    
    void* args[] = {
        &d_input, &d_output, &d_weight,
        &u_in_h, &u_in_w, &u_kh, &u_kw,
        &u_num_filters, &u_out_h, &u_out_w, &N
    };
    
    PTX_CHECK(cuLaunchKernel(conv2d_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// conv2d_multi_channel: Multi-channel 2D convolution
// input[in_c][in_h][in_w] * weight[out_c][in_c][kh][kw] -> output[out_c][out_h][out_w]
void launch_conv2d_mc_ptx(float* d_input, float* d_output, float* d_weight,
                          int in_c, int in_h, int in_w, int kh, int kw, int out_c,
                          int block_size)
{
    if (conv2d_mc_kernel == nullptr) {
        fprintf(stderr, "Error: PTX forward not initialized. Call init_forward_ptx() first.\n");
        return;
    }
    
    int out_h = in_h - kh + 1;
    int out_w = in_w - kw + 1;
    unsigned int N = kh * kw * in_c * out_c * out_h * out_w;
    int grid_size = (N + block_size - 1) / block_size;
    
    unsigned int u_in_c = in_c, u_in_h = in_h, u_in_w = in_w;
    unsigned int u_kh = kh, u_kw = kw, u_out_c = out_c;
    unsigned int u_out_h = out_h, u_out_w = out_w;
    
    void* args[] = {
        &d_input, &d_output, &d_weight,
        &u_in_c, &u_in_h, &u_in_w, &u_kh, &u_kw,
        &u_out_c, &u_out_h, &u_out_w, &N
    };
    
    PTX_CHECK(cuLaunchKernel(conv2d_mc_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// add_bias: Add per-channel bias
// output[c][h][w] += bias[c]
void launch_add_bias_ptx(float* d_data, float* d_bias, int channels, int spatial_size, int block_size)
{
    if (add_bias_kernel == nullptr) {
        fprintf(stderr, "Error: PTX forward not initialized. Call init_forward_ptx() first.\n");
        return;
    }
    
    unsigned int N = channels * spatial_size;
    int grid_size = (N + block_size - 1) / block_size;
    
    unsigned int u_channels = channels, u_spatial = spatial_size;
    
    void* args[] = { &d_data, &d_bias, &u_channels, &u_spatial, &N };
    
    PTX_CHECK(cuLaunchKernel(add_bias_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// add_bias_shared: Add single shared bias to all elements
// output[i] += bias[0] for all i
void launch_add_bias_shared_ptx(float* d_data, float* d_bias, int n, int block_size)
{
    if (add_bias_shared_kernel == nullptr) {
        fprintf(stderr, "Error: PTX forward not initialized. Call init_forward_ptx() first.\n");
        return;
    }
    
    unsigned int N = n;
    int grid_size = (N + block_size - 1) / block_size;
    
    void* args[] = { &d_data, &d_bias, &N };
    
    PTX_CHECK(cuLaunchKernel(add_bias_shared_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// pooling: General weighted pooling (subsampling)
// input[c][in_h][in_w] * weight[kh][kw] -> output[c][out_h][out_w]
void launch_pooling_ptx(float* d_input, float* d_output, float* d_weight,
                        int channels, int in_h, int in_w, int kh, int kw,
                        int block_size)
{
    if (pooling_kernel == nullptr) {
        fprintf(stderr, "Error: PTX forward not initialized. Call init_forward_ptx() first.\n");
        return;
    }
    
    int out_h = in_h / kh;
    int out_w = in_w / kw;
    unsigned int N = kh * kw * channels * out_h * out_w;
    int grid_size = (N + block_size - 1) / block_size;
    
    unsigned int u_channels = channels, u_in_h = in_h, u_in_w = in_w;
    unsigned int u_kh = kh, u_kw = kw, u_out_h = out_h, u_out_w = out_w;
    
    void* args[] = {
        &d_input, &d_output, &d_weight,
        &u_channels, &u_in_h, &u_in_w, &u_kh, &u_kw, &u_out_h, &u_out_w, &N
    };
    
    PTX_CHECK(cuLaunchKernel(pooling_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// fc_forward: Fully connected layer
// input[in_size] * weight[out_size][in_size] -> output[out_size]
void launch_fc_forward_ptx(float* d_input, float* d_output, float* d_weight,
                           int in_size, int out_size, int block_size)
{
    if (fc_forward_kernel == nullptr) {
        fprintf(stderr, "Error: PTX forward not initialized. Call init_forward_ptx() first.\n");
        return;
    }
    
    unsigned int N = in_size * out_size;
    int grid_size = (N + block_size - 1) / block_size;
    
    unsigned int u_in_size = in_size, u_out_size = out_size;
    
    void* args[] = { &d_input, &d_output, &d_weight, &u_in_size, &u_out_size, &N };
    
    PTX_CHECK(cuLaunchKernel(fc_forward_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// apply_grad: output[i] += learning_rate * grad[i]
void launch_apply_grad_ptx(float* d_output, float* d_grad, float learning_rate, int n, int block_size)
{
    if (apply_grad_kernel == nullptr) {
        fprintf(stderr, "Error: PTX forward not initialized. Call init_forward_ptx() first.\n");
        return;
    }
    
    int grid_size = (n + block_size - 1) / block_size;
    unsigned int n_u32 = static_cast<unsigned int>(n);
    
    void* args[] = { &d_output, &d_grad, &learning_rate, &n_u32 };
    
    PTX_CHECK(cuLaunchKernel(apply_grad_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// memset_zero: Zero out a buffer
void launch_memset_zero_ptx(float* d_data, int n, int block_size)
{
    if (memset_zero_kernel == nullptr) {
        fprintf(stderr, "Error: PTX forward not initialized. Call init_forward_ptx() first.\n");
        return;
    }
    
    unsigned int N = n;
    int grid_size = (N + block_size - 1) / block_size;
    
    void* args[] = { &d_data, &N };
    
    PTX_CHECK(cuLaunchKernel(memset_zero_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// ============================================================================
// General Backward Pass PTX Launch Functions
// ============================================================================

// bp_sigmoid_grad: d_preact[i] = d_output[i] * sigmoid(preact[i]) * (1 - sigmoid(preact[i]))
void launch_bp_sigmoid_grad_ptx(float* d_preact, float* d_output, float* preact, int n, int block_size)
{
    if (bp_sigmoid_grad_kernel == nullptr) {
        fprintf(stderr, "Error: PTX backward not initialized. Call init_backward_ptx() first.\n");
        return;
    }
    
    unsigned int N = n;
    int grid_size = (N + block_size - 1) / block_size;
    
    void* args[] = { &d_preact, &d_output, &preact, &N };
    
    PTX_CHECK(cuLaunchKernel(bp_sigmoid_grad_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// bp_fc_weight: d_weight[out * in_size + in] = d_preact[out] * prev_output[in]
void launch_bp_fc_weight_ptx(float* d_weight, float* d_preact, float* prev_output,
                             int in_size, int out_size, int block_size)
{
    if (bp_fc_weight_kernel == nullptr) {
        fprintf(stderr, "Error: PTX backward not initialized. Call init_backward_ptx() first.\n");
        return;
    }
    
    unsigned int N = in_size * out_size;
    int grid_size = (N + block_size - 1) / block_size;
    
    unsigned int u_in_size = in_size, u_out_size = out_size;
    
    void* args[] = { &d_weight, &d_preact, &prev_output, &u_in_size, &u_out_size, &N };
    
    PTX_CHECK(cuLaunchKernel(bp_fc_weight_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// bp_fc_bias: bias[i] += dt * d_preact[i]
void launch_bp_fc_bias_ptx(float* bias, float* d_preact, float dt, int n, int block_size)
{
    if (bp_fc_bias_kernel == nullptr) {
        fprintf(stderr, "Error: PTX backward not initialized. Call init_backward_ptx() first.\n");
        return;
    }
    
    unsigned int N = n;
    int grid_size = (N + block_size - 1) / block_size;
    
    void* args[] = { &bias, &d_preact, &dt, &N };
    
    PTX_CHECK(cuLaunchKernel(bp_fc_bias_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// bp_fc_output: d_output[in] += weight[out * in_size + in] * d_preact[out] (atomicAdd)
void launch_bp_fc_output_ptx(float* d_output, float* weight, float* d_preact,
                             int in_size, int out_size, int block_size)
{
    if (bp_fc_output_kernel == nullptr) {
        fprintf(stderr, "Error: PTX backward not initialized. Call init_backward_ptx() first.\n");
        return;
    }
    
    unsigned int N = in_size * out_size;
    int grid_size = (N + block_size - 1) / block_size;
    
    unsigned int u_in_size = in_size, u_out_size = out_size;
    
    void* args[] = { &d_output, &weight, &d_preact, &u_in_size, &u_out_size, &N };
    
    PTX_CHECK(cuLaunchKernel(bp_fc_output_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// bp_pooling_weight: Compute pooling weight gradient
void launch_bp_pooling_weight_ptx(float* d_weight, float* d_preact, float* prev_output,
                                  int channels, int in_h, int in_w, int kh, int kw,
                                  int block_size)
{
    if (bp_pooling_weight_kernel == nullptr) {
        fprintf(stderr, "Error: PTX backward not initialized. Call init_backward_ptx() first.\n");
        return;
    }
    
    int out_h = in_h / kh;
    int out_w = in_w / kw;
    unsigned int N = kh * kw * channels * out_h * out_w;
    int grid_size = (N + block_size - 1) / block_size;
    
    float norm = static_cast<float>(channels * out_h * out_w);
    unsigned int u_channels = channels, u_in_h = in_h, u_in_w = in_w;
    unsigned int u_kh = kh, u_kw = kw, u_out_h = out_h, u_out_w = out_w;
    
    void* args[] = {
        &d_weight, &d_preact, &prev_output,
        &u_channels, &u_in_h, &u_in_w, &u_kh, &u_kw, &u_out_h, &u_out_w,
        &norm, &N
    };
    
    PTX_CHECK(cuLaunchKernel(bp_pooling_weight_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// bp_pooling_bias_shared: bias[0] += dt * sum(d_preact) / n
void launch_bp_pooling_bias_shared_ptx(float* bias, float* d_preact, float dt, int n, int block_size)
{
    if (bp_pooling_bias_shared_kernel == nullptr) {
        fprintf(stderr, "Error: PTX backward not initialized. Call init_backward_ptx() first.\n");
        return;
    }
    
    unsigned int N = n;
    int grid_size = (N + block_size - 1) / block_size;
    
    void* args[] = { &bias, &d_preact, &dt, &N };
    
    PTX_CHECK(cuLaunchKernel(bp_pooling_bias_shared_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// bp_pooling_output: Backprop through pooling layer
void launch_bp_pooling_output_ptx(float* d_output, float* weight, float* d_preact,
                                  int channels, int in_h, int in_w, int kh, int kw,
                                  int block_size)
{
    if (bp_pooling_output_kernel == nullptr) {
        fprintf(stderr, "Error: PTX backward not initialized. Call init_backward_ptx() first.\n");
        return;
    }
    
    int out_h = in_h / kh;
    int out_w = in_w / kw;
    unsigned int N = kh * kw * channels * out_h * out_w;
    int grid_size = (N + block_size - 1) / block_size;
    
    unsigned int u_channels = channels, u_in_h = in_h, u_in_w = in_w;
    unsigned int u_kh = kh, u_kw = kw, u_out_h = out_h, u_out_w = out_w;
    
    void* args[] = {
        &d_output, &weight, &d_preact,
        &u_channels, &u_in_h, &u_in_w, &u_kh, &u_kw, &u_out_h, &u_out_w, &N
    };
    
    PTX_CHECK(cuLaunchKernel(bp_pooling_output_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// bp_conv2d_weight: Compute conv weight gradient
void launch_bp_conv2d_weight_ptx(float* d_weight, float* d_preact, float* prev_output,
                                 int in_h, int in_w, int kh, int kw, int num_filters,
                                 int block_size)
{
    if (bp_conv2d_weight_kernel == nullptr) {
        fprintf(stderr, "Error: PTX backward not initialized. Call init_backward_ptx() first.\n");
        return;
    }
    
    int out_h = in_h - kh + 1;
    int out_w = in_w - kw + 1;
    unsigned int N = num_filters * kh * kw * out_h * out_w;
    int grid_size = (N + block_size - 1) / block_size;
    
    float norm = static_cast<float>(out_h * out_w);
    unsigned int u_in_h = in_h, u_in_w = in_w, u_kh = kh, u_kw = kw;
    unsigned int u_num_filters = num_filters, u_out_h = out_h, u_out_w = out_w;
    
    void* args[] = {
        &d_weight, &d_preact, &prev_output,
        &u_in_h, &u_in_w, &u_kh, &u_kw, &u_num_filters, &u_out_h, &u_out_w,
        &norm, &N
    };
    
    PTX_CHECK(cuLaunchKernel(bp_conv2d_weight_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}

// bp_conv2d_bias: Per-channel bias update with normalization
void launch_bp_conv2d_bias_ptx(float* bias, float* d_preact, float dt,
                               int num_filters, int out_h, int out_w, int block_size)
{
    if (bp_conv2d_bias_kernel == nullptr) {
        fprintf(stderr, "Error: PTX backward not initialized. Call init_backward_ptx() first.\n");
        return;
    }
    
    unsigned int N = num_filters * out_h * out_w;
    int grid_size = (N + block_size - 1) / block_size;
    
    float norm = static_cast<float>(out_h * out_w);
    unsigned int u_num_filters = num_filters, u_out_h = out_h, u_out_w = out_w;
    
    void* args[] = { &bias, &d_preact, &dt, &u_num_filters, &u_out_h, &u_out_w, &norm, &N };
    
    PTX_CHECK(cuLaunchKernel(bp_conv2d_bias_kernel, grid_size, 1, 1, block_size, 1, 1, 0, 0, args, nullptr));
}
