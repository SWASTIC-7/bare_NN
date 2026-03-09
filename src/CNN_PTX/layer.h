#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer {
	public:
	int M, N, O;   //Dimensions of the layers\
                   // M = number of output neurons\
                   // N = number of input neurons\
                   // O = number of filters (for conv layers)   

	float *output;
	float *preact;

	float *bias;
	float *weight;

	float *d_output;
	float *d_preact;
	float *d_weight;

	Layer(int M, int N, int O);

	~Layer();

	void setOutput(float *data);
	void clear();
	void bp_clear();
};

// PTX-based activation functions (loaded from activation_fn.ptx)
void init_activation_ptx(const char* ptx_path);
void cleanup_activation_ptx();
void launch_sigmoid_ptx(float* d_input, float* d_output, int n, int block_size = 256);

// PTX-based loss functions (loaded from losses.ptx)
void init_loss_ptx(const char* ptx_path);
void cleanup_loss_ptx();
void launch_make_error_ptx(float* d_error, float* d_output, unsigned int label, int n, int block_size = 256);
void launch_mse_gradient_ptx(float* d_predicted, float* d_target, float* d_gradient, int n, int block_size = 256);

// PTX-based forward pass functions (loaded from forward_pass.ptx)
void init_forward_ptx(const char* ptx_path);
void cleanup_forward_ptx();

// General 2D convolution (single-channel input -> multi-filter output)
void launch_conv2d_ptx(float* d_input, float* d_output, float* d_weight,
                       int in_h, int in_w, int kh, int kw, int num_filters,
                       int block_size = 256);

// Multi-channel 2D convolution
void launch_conv2d_mc_ptx(float* d_input, float* d_output, float* d_weight,
                          int in_c, int in_h, int in_w, int kh, int kw, int out_c,
                          int block_size = 256);

// Per-channel bias: output[c][h][w] += bias[c]
void launch_add_bias_ptx(float* d_data, float* d_bias, int channels, int spatial_size,
                         int block_size = 256);

// Shared bias: output[i] += bias[0] for all i
void launch_add_bias_shared_ptx(float* d_data, float* d_bias, int n, int block_size = 256);

// Weighted pooling/subsampling
void launch_pooling_ptx(float* d_input, float* d_output, float* d_weight,
                        int channels, int in_h, int in_w, int kh, int kw,
                        int block_size = 256);

// Fully connected layer
void launch_fc_forward_ptx(float* d_input, float* d_output, float* d_weight,
                           int in_size, int out_size, int block_size = 256);

// Gradient application: output[i] += learning_rate * grad[i]
void launch_apply_grad_ptx(float* d_output, float* d_grad, float learning_rate, int n,
                           int block_size = 256);

// Zero buffer
void launch_memset_zero_ptx(float* d_data, int n, int block_size = 256);

// PTX-based backward pass functions (loaded from backward_pass.ptx)
void init_backward_ptx(const char* ptx_path);
void cleanup_backward_ptx();

// Sigmoid gradient: d_preact[i] = d_output[i] * sigmoid(preact[i]) * (1 - sigmoid(preact[i]))
void launch_bp_sigmoid_grad_ptx(float* d_preact, float* d_output, float* preact, int n,
                                int block_size = 256);

// FC weight gradient: d_weight[out * in_size + in] = d_preact[out] * prev_output[in]
void launch_bp_fc_weight_ptx(float* d_weight, float* d_preact, float* prev_output,
                             int in_size, int out_size, int block_size = 256);

// FC bias update: bias[i] += dt * d_preact[i]
void launch_bp_fc_bias_ptx(float* bias, float* d_preact, float dt, int n, int block_size = 256);

// FC backprop to previous layer: d_output[in] += weight * d_preact
void launch_bp_fc_output_ptx(float* d_output, float* weight, float* d_preact,
                             int in_size, int out_size, int block_size = 256);

// Pooling weight gradient
void launch_bp_pooling_weight_ptx(float* d_weight, float* d_preact, float* prev_output,
                                  int channels, int in_h, int in_w, int kh, int kw,
                                  int block_size = 256);

// Pooling shared bias update
void launch_bp_pooling_bias_shared_ptx(float* bias, float* d_preact, float dt, int n,
                                       int block_size = 256);

// Pooling backprop to previous layer
void launch_bp_pooling_output_ptx(float* d_output, float* weight, float* d_preact,
                                  int channels, int in_h, int in_w, int kh, int kw,
                                  int block_size = 256);

// Conv weight gradient
void launch_bp_conv2d_weight_ptx(float* d_weight, float* d_preact, float* prev_output,
                                 int in_h, int in_w, int kh, int kw, int num_filters,
                                 int block_size = 256);

// Conv per-channel bias update
void launch_bp_conv2d_bias_ptx(float* bias, float* d_preact, float dt,
                               int num_filters, int out_h, int out_w, int block_size = 256);