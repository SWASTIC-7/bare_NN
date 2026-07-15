// CUDA implementation of the thread-tiling pseudocode from the blog:
//   block tiling into shared memory  +  register (thread) tiling via an outer product.
//
// The tile sizes are compile-time constants (register arrays need a known size),
// overridable with -D at compile time. This is the CUDA counterpart of new.ptx.
//
// Standalone build:   nvcc -O3 thread_tiled.cu -o tt && ./tt
// The benchmark harness (run_bench.py) compiles just the kernel via NVRTC and
// sweeps the tile sizes; the main() below is excluded under NVRTC.

#ifdef __CUDACC_RTC__
typedef unsigned int uint;          // NVRTC doesn't define `uint` for us
#endif

#ifndef BM
#define BM 64          // block tile rows (M)
#endif
#ifndef BN
#define BN 64          // block tile cols (N)
#endif
#ifndef BK
#define BK 8           // block tile depth (K)
#endif
#ifndef TM
#define TM 4           // thread tile rows  (== TT_Y)
#endif
#ifndef TN
#define TN 4           // thread tile cols  (== TT_X)
#endif

extern "C" __global__ void thread_tiled_matmul(
        int M, int N, int K,
        const float *A, const float *B, float *C)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;

    // this thread owns a TM x TN block of C inside the BM x BN block tile
    const uint threads_per_row = BN / TN;
    const uint inner_row = threadIdx.x / threads_per_row;
    const uint inner_col = threadIdx.x % threads_per_row;

    __shared__ float sh_A[BM * BK];
    __shared__ float sh_B[BK * BN];

    // advance global pointers to this block's tile
    A += block_row * BM * K;
    B += block_col * BN;
    C += block_row * BM * N + block_col * BN;

    // cooperative-load coordinates (flat thread index, strided over the tile)
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA   = blockDim.x / BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB   = blockDim.x / BN;

    float value[TM * TN] = {0.0f};   // accumulators, live in registers
    float reg_A[TM];
    float reg_B[TN];

    for (uint bk = 0; bk < K; bk += BK)
    {
        // --- load the BM x BK and BK x BN tiles into shared memory (once) ---
        for (uint off = 0; off < BM; off += strideA)
            sh_A[(innerRowA + off) * BK + innerColA] =
                A[(innerRowA + off) * K + innerColA];
        for (uint off = 0; off < BK; off += strideB)
            sh_B[(innerRowB + off) * BN + innerColB] =
                B[(innerRowB + off) * N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        // --- register-tiled outer product ---
        for (uint dot = 0; dot < BK; dot++)
        {
            // load one strip of A and one strip of B into registers, ONCE
            for (uint i = 0; i < TM; i++)
                reg_A[i] = sh_A[(inner_row * TM + i) * BK + dot];
            for (uint i = 0; i < TN; i++)
                reg_B[i] = sh_B[dot * BN + inner_col * TN + i];

            // reuse them across TM*TN multiply-adds
            for (uint rr = 0; rr < TM; rr++)
                for (uint cc = 0; cc < TN; cc++)
                    value[rr * TN + cc] += reg_A[rr] * reg_B[cc];
        }
        __syncthreads();
    }

    for (uint rr = 0; rr < TM; rr++)
        for (uint cc = 0; cc < TN; cc++)
            C[(inner_row * TM + rr) * N + inner_col * TN + cc] = value[rr * TN + cc];
}

#ifndef __CUDACC_RTC__
// ------------------------- standalone benchmark -------------------------
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main()
{
    const int M = 1024, N = 1024, K = 1024;
    size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;

    float *A, *B, *C;
    cudaMallocManaged(&A, szA * sizeof(float));
    cudaMallocManaged(&B, szB * sizeof(float));
    cudaMallocManaged(&C, szC * sizeof(float));
    for (size_t i = 0; i < szA; i++) A[i] = (float)(rand() % 5 - 2);
    for (size_t i = 0; i < szB; i++) B[i] = (float)(rand() % 5 - 2);

    dim3 block((BM * BN) / (TM * TN));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    thread_tiled_matmul<<<grid, block>>>(M, N, K, A, B, C);   // warmup
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    const int iters = 50;
    cudaEventRecord(s);
    for (int it = 0; it < iters; it++)
        thread_tiled_matmul<<<grid, block>>>(M, N, K, A, B, C);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms, s, e);
    ms /= iters;
    double gf = (2.0 * M * N * K) / (ms * 1e-3) / 1e9;
    printf("thread_tiled_matmul  BM=%d BN=%d BK=%d TM=%d TN=%d : %.3f ms, %.1f GFLOP/s\n",
           BM, BN, BK, TM, TN, ms, gf);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
#endif
