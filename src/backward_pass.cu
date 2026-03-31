#include "bare_nn.h"
#include "ptx_dispatch.cuh"

namespace {

bare_nn::PtxKernelLibrary& lib() {
	static bare_nn::PtxKernelLibrary kLib;
	return kLib;
}

}  // namespace

namespace bare_nn {

void mse_gradient(
	CUdeviceptr d_predicted,
	CUdeviceptr d_target,
	CUdeviceptr d_gradient,
	unsigned int n,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_predicted, &d_target, &d_gradient, &n};
	lib().launch1D("losses_grad.ptx", "mse_gradient", static_cast<int>(n), block_size, args, stream);
}

void l2_gradient(
	CUdeviceptr d_predicted,
	CUdeviceptr d_target,
	CUdeviceptr d_gradient,
	unsigned int n,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_predicted, &d_target, &d_gradient, &n};
	lib().launch1D("losses_grad.ptx", "l2_gradient", static_cast<int>(n), block_size, args, stream);
}

void relu_grad(
	CUdeviceptr d_input,
	CUdeviceptr d_grad_in,
	CUdeviceptr d_grad_out,
	unsigned int n,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_input, &d_grad_in, &d_grad_out, &n};
	lib().launch1D("activation_grad.ptx", "relu_grad", static_cast<int>(n), block_size, args, stream);
}

void sigmoid_grad(
	CUdeviceptr d_sigmoid_out,
	CUdeviceptr d_grad_in,
	CUdeviceptr d_grad_out,
	unsigned int n,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_sigmoid_out, &d_grad_in, &d_grad_out, &n};
	lib().launch1D("activation_grad.ptx", "sigmoid_grad", static_cast<int>(n), block_size, args, stream);
}

void sigmoid_grad_from_input(
	CUdeviceptr d_input,
	CUdeviceptr d_grad_in,
	CUdeviceptr d_grad_out,
	unsigned int n,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_input, &d_grad_in, &d_grad_out, &n};
	lib().launch1D("activation_grad.ptx", "sigmoid_grad_from_input", static_cast<int>(n), block_size, args, stream);
}

void tanh_grad(
	CUdeviceptr d_tanh_out,
	CUdeviceptr d_grad_in,
	CUdeviceptr d_grad_out,
	unsigned int n,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_tanh_out, &d_grad_in, &d_grad_out, &n};
	lib().launch1D("activation_grad.ptx", "tanh_grad", static_cast<int>(n), block_size, args, stream);
}

void tanh_grad_from_input(
	CUdeviceptr d_input,
	CUdeviceptr d_grad_in,
	CUdeviceptr d_grad_out,
	unsigned int n,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_input, &d_grad_in, &d_grad_out, &n};
	lib().launch1D("activation_grad.ptx", "tanh_grad_from_input", static_cast<int>(n), block_size, args, stream);
}

void leaky_relu_grad(
	CUdeviceptr d_input,
	CUdeviceptr d_grad_in,
	CUdeviceptr d_grad_out,
	unsigned int n,
	float alpha,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_input, &d_grad_in, &d_grad_out, &alpha, &n};
	lib().launch1D("activation_grad.ptx", "leaky_relu_grad", static_cast<int>(n), block_size, args, stream);
}

void gelu_grad(
	CUdeviceptr d_input,
	CUdeviceptr d_grad_in,
	CUdeviceptr d_grad_out,
	unsigned int n,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_input, &d_grad_in, &d_grad_out, &n};
	lib().launch1D("activation_grad.ptx", "gelu_grad", static_cast<int>(n), block_size, args, stream);
}

void relu_grad_vec4(
	CUdeviceptr d_input,
	CUdeviceptr d_grad_in,
	CUdeviceptr d_grad_out,
	unsigned int n,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_input, &d_grad_in, &d_grad_out, &n};
	lib().launch1D("activation_grad.ptx", "relu_grad_vec4", static_cast<int>(n), block_size, args, stream);
}

void sigmoid_grad_vec4(
	CUdeviceptr d_sigmoid_out,
	CUdeviceptr d_grad_in,
	CUdeviceptr d_grad_out,
	unsigned int n,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_sigmoid_out, &d_grad_in, &d_grad_out, &n};
	lib().launch1D("activation_grad.ptx", "sigmoid_grad_vec4", static_cast<int>(n), block_size, args, stream);
}

void softmax_grad(CUdeviceptr d_softmax_out, CUdeviceptr d_grad_in, CUdeviceptr d_grad_out, unsigned int n) {
	(void)d_softmax_out;
	(void)d_grad_in;
	(void)d_grad_out;
	(void)n;
	// TODO: add ptx for this
}

void fully_connected_backward() {
	// TODO: add ptx for this
}

}  // namespace bare_nn

