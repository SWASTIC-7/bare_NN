#include "bare_nn.h"
#include "ptx_dispatch.cuh"

namespace {

bare_nn::PtxKernelLibrary& lib() {
	static bare_nn::PtxKernelLibrary kLib;
	return kLib;
}

}  // namespace

namespace bare_nn {

void vector_add(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_out, unsigned int n, int block_size, CUstream stream) {
	void* args[] = {&d_a, &d_b, &d_out, &n};
	lib().launch1D("vector_operation.ptx", "vectorAdd", static_cast<int>(n), block_size, args, stream);
}

void vector_sub(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_out, unsigned int n, int block_size, CUstream stream) {
	void* args[] = {&d_a, &d_b, &d_out, &n};
	lib().launch1D("vector_operation.ptx", "vectorSub", static_cast<int>(n), block_size, args, stream);
}

void vector_mul(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_out, unsigned int n, int block_size, CUstream stream) {
	void* args[] = {&d_a, &d_b, &d_out, &n};
	lib().launch1D("vector_operation.ptx", "vectorMul", static_cast<int>(n), block_size, args, stream);
}

void vector_div(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_out, unsigned int n, int block_size, CUstream stream) {
	void* args[] = {&d_a, &d_b, &d_out, &n};
	lib().launch1D("vector_operation.ptx", "vectorDiv", static_cast<int>(n), block_size, args, stream);
}

void vector_scalar_add(CUdeviceptr d_a, float scalar, CUdeviceptr d_out, unsigned int n, int block_size, CUstream stream) {
	void* args[] = {&d_a, &scalar, &d_out, &n};
	lib().launch1D("vector_operation.ptx", "vectorScalarAdd", static_cast<int>(n), block_size, args, stream);
}

void vector_scalar_mul(CUdeviceptr d_a, float scalar, CUdeviceptr d_out, unsigned int n, int block_size, CUstream stream) {
	void* args[] = {&d_a, &scalar, &d_out, &n};
	lib().launch1D("vector_operation.ptx", "vectorScalarMul", static_cast<int>(n), block_size, args, stream);
}

void reduce_sum(CUdeviceptr d_input, CUdeviceptr d_out_scalar, unsigned int n, CUstream stream) {
	// Current PTX reduction uses one block and in-place reduction on input buffer.
	void* args[] = {&d_input, &d_out_scalar, &n};
	CUfunction fn = lib().getFunction("reduction_op.ptx", "sum");
	CU_CHECK(cuLaunchKernel(fn, 1, 1, 1, n, 1, 1, 0, stream, args, nullptr));
}

void reduce_mean(CUdeviceptr d_input, CUdeviceptr d_out_scalar, unsigned int n, CUstream stream) {
	void* args[] = {&d_input, &d_out_scalar, &n};
	CUfunction fn = lib().getFunction("reduction_op.ptx", "mean");
	CU_CHECK(cuLaunchKernel(fn, 1, 1, 1, n, 1, 1, 0, stream, args, nullptr));
}

void reduce_max(CUdeviceptr d_input, CUdeviceptr d_out_scalar, unsigned int n, CUstream stream) {
	void* args[] = {&d_input, &d_out_scalar, &n};
	CUfunction fn = lib().getFunction("reduction_op.ptx", "max");
	CU_CHECK(cuLaunchKernel(fn, 1, 1, 1, n, 1, 1, 0, stream, args, nullptr));
}

void reduce_min(CUdeviceptr d_input, CUdeviceptr d_out_scalar, unsigned int n, CUstream stream) {
	void* args[] = {&d_input, &d_out_scalar, &n};
	CUfunction fn = lib().getFunction("reduction_op.ptx", "min");
	CU_CHECK(cuLaunchKernel(fn, 1, 1, 1, n, 1, 1, 0, stream, args, nullptr));
}

void matmul(
	CUdeviceptr d_c,
	CUdeviceptr d_a,
	CUdeviceptr d_b,
	unsigned int M,
	unsigned int N,
	unsigned int K,
	const MatmulConfig& config,
	CUstream stream) {
	const dim3 block((config.BT_M / config.TT_Y) * (config.BT_N / config.TT_X), 1, 1);
	const dim3 grid((N + config.BT_N - 1) / config.BT_N, (M + config.BT_M - 1) / config.BT_M, 1);
	const unsigned int shared_mem_bytes = (config.BT_M * config.BT_K + config.BT_K * config.BT_N) * sizeof(float);

	unsigned int bt_m = config.BT_M;
	unsigned int bt_n = config.BT_N;
	unsigned int bt_k = config.BT_K;
	unsigned int wt_x = config.WT_X;
	unsigned int wt_y = config.WT_Y;
	unsigned int tt_x = config.TT_X;
	unsigned int tt_y = config.TT_Y;

	void* args[] = {
		&d_c,
		&d_a,
		&d_b,
		&M,
		&N,
		&K,
		&bt_m,
		&bt_n,
		&bt_k,
		&wt_x,
		&wt_y,
		&tt_x,
		&tt_y
	};

	lib().launch2D("matmul.ptx", "matmul", grid, block, args, stream, shared_mem_bytes);
}

void matmul_complex(
	CUdeviceptr d_c,
	CUdeviceptr d_a,
	CUdeviceptr d_b,
	unsigned int M,
	unsigned int N,
	unsigned int K,
	int block_size,
	CUstream stream) {
	const unsigned int total = M * N;
	void* args[] = {&d_c, &d_a, &d_b, &M, &N, &K};
	lib().launch1D("matmul.ptx", "ptx_mul_matrix_complex", static_cast<int>(total), block_size, args, stream);
}

void fully_connected_forward(
	CUdeviceptr d_input,
	CUdeviceptr d_weight,
	CUdeviceptr d_output,
	unsigned int in_size,
	unsigned int out_size,
	const MatmulConfig& config,
	CUstream stream) {
	// Uses matmul PTX: [1 x in_size] * [in_size x out_size] -> [1 x out_size].
	matmul(d_output, d_input, d_weight, 1, out_size, in_size, config, stream);
}

void conv2d_forward() {
	// TODO: add ptx for this
}

}  // namespace bare_nn

