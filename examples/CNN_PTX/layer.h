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