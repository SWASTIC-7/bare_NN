#include "bare_nn.h"
#include "ptx_dispatch.cuh"

namespace {

bare_nn::PtxKernelLibrary& lib() {
	static bare_nn::PtxKernelLibrary kLib;
	return kLib;
}

}  // namespace

namespace bare_nn {

void l2_squared_diff(
	CUdeviceptr d_predicted,
	CUdeviceptr d_target,
	CUdeviceptr d_squared_diff,
	unsigned int n,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_predicted, &d_target, &d_squared_diff, &n};
	lib().launch1D("losses.ptx", "l2_squared_diff", static_cast<int>(n), block_size, args, stream);
}

void l2_norm_reduce(
	CUdeviceptr d_predicted,
	CUdeviceptr d_target,
	CUdeviceptr d_partial_sums,
	unsigned int n,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_predicted, &d_target, &d_partial_sums, &n};
	lib().launch1D("losses.ptx", "l2_norm_reduce", static_cast<int>(n), block_size, args, stream);
}

void mse_reduce(
	CUdeviceptr d_predicted,
	CUdeviceptr d_target,
	CUdeviceptr d_partial_sums,
	unsigned int n,
	int block_size,
	CUstream stream) {
	void* args[] = {&d_predicted, &d_target, &d_partial_sums, &n};
	lib().launch1D("losses.ptx", "mse_reduce", static_cast<int>(n), block_size, args, stream);
}

}  // namespace bare_nn

