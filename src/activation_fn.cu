#include "bare_nn.h"
#include "ptx_dispatch.cuh"

#include <stdio.h>

namespace {

bare_nn::PtxKernelLibrary& lib() {
	static bare_nn::PtxKernelLibrary kLib;
	return kLib;
}

unsigned int next_pow2(unsigned int x) {
	if (x <= 1) {
		return 1;
	}
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x + 1;
}

}  // namespace

namespace bare_nn {

void relu(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n, int block_size, CUstream stream) {
	void* args[] = {&d_input, &d_output, &n};
	lib().launch1D("activation_fn.ptx", "relu", static_cast<int>(n), block_size, args, stream);
}

void sigmoid(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n, int block_size, CUstream stream) {
	void* args[] = {&d_input, &d_output, &n};
	lib().launch1D("activation_fn.ptx", "sigmoid", static_cast<int>(n), block_size, args, stream);
}

void tanh_activation(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n, int block_size, CUstream stream) {
	void* args[] = {&d_input, &d_output, &n};
	lib().launch1D("activation_fn.ptx", "tanh", static_cast<int>(n), block_size, args, stream);
}

void softmax(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n, int block_size, CUstream stream) {
	// This PTX kernel is single-block and uses shared memory reduction.
	if (n > 1024) {
		fprintf(stderr, "softmax currently supports n <= 1024 (got %u)\n", n);
		exit(EXIT_FAILURE);
	}

	const unsigned int launch_block = next_pow2(n);
	const unsigned int threads = launch_block > 1024 ? 1024 : launch_block;

	void* args[] = {&d_input, &d_output, &n};
	CUfunction fn = lib().getFunction("activation_fn.ptx", "softmax");
	CU_CHECK(cuLaunchKernel(fn, 1, 1, 1, threads, 1, 1, 0, stream, args, nullptr));
}

void log_softmax(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n, int block_size, CUstream stream) {
	(void)block_size;
	if (n > 1024) {
		fprintf(stderr, "log_softmax currently supports n <= 1024 (got %u)\n", n);
		exit(EXIT_FAILURE);
	}

	const unsigned int launch_block = next_pow2(n);
	const unsigned int threads = launch_block > 1024 ? 1024 : launch_block;

	void* args[] = {&d_input, &d_output, &n};
	CUfunction fn = lib().getFunction("activation_fn.ptx", "log_softmax");
	CU_CHECK(cuLaunchKernel(fn, 1, 1, 1, threads, 1, 1, 0, stream, args, nullptr));
}

void leaky_relu(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n, float alpha) {
	(void)d_input;
	(void)d_output;
	(void)n;
	(void)alpha;
	// TODO: add ptx for this
}

void gelu(CUdeviceptr d_input, CUdeviceptr d_output, unsigned int n) {
	(void)d_input;
	(void)d_output;
	(void)n;
	// TODO: add ptx for this
}

}  // namespace bare_nn

