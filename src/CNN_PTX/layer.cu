#include "layer.h"
#include <cstdio>

// PTX module and kernel handles for activation functions
static CUmodule activation_module = nullptr;
static CUfunction sigmoid_kernel = nullptr;

// PTX module and kernel handles for loss functions
static CUmodule loss_module = nullptr;
static CUfunction make_error_onehot_kernel = nullptr;
static CUfunction mse_gradient_kernel = nullptr;

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
