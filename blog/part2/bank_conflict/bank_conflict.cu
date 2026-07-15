// Vectorized CUDA thread-tiled matmul + shared-memory bank-conflict padding.
// Same as vectorized.cu, but the transposed A tile leading dim is padded to LDA = BM+4
// so a column stride is no longer a multiple of 32 banks (kills the transpose-store conflict).
// Assumes M,N,K multiples of the block tile and of 4.  Tile sizes are compile-time (-D).

#ifdef __CUDACC_RTC__
typedef unsigned int uint;
#endif

#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 16
#endif
#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 4
#endif
#define LDA (BM + 4)                    // padded leading dim of transposed sh_A

extern "C" __global__ void thread_tiled_matmul(
        int M, int N, int K,
        const float *A, const float *B, float *C)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;
    const uint threads_per_row = BN / TN;
    const uint inner_row = threadIdx.x / threads_per_row;
    const uint inner_col = threadIdx.x % threads_per_row;

    __shared__ __align__(16) float sh_A[BK * LDA];   // transposed + padded: [BK][LDA]
    __shared__ __align__(16) float sh_B[BK * BN];

    A += block_row * BM * K;
    B += block_col * BN;
    C += block_row * BM * N + block_col * BN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint strideA   = blockDim.x / (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    const uint strideB   = blockDim.x / (BN / 4);

    float value[TM * TN] = {0.0f};
    float reg_A[TM];
    float reg_B[TN];

    for (uint bk = 0; bk < K; bk += BK)
    {
        for (uint off = 0; off < BM; off += strideA)
        {
            float4 tmp = *reinterpret_cast<const float4*>(
                &A[(innerRowA + off) * K + innerColA * 4]);
            sh_A[(innerColA * 4 + 0) * LDA + innerRowA + off] = tmp.x;   // stride LDA, not BM
            sh_A[(innerColA * 4 + 1) * LDA + innerRowA + off] = tmp.y;
            sh_A[(innerColA * 4 + 2) * LDA + innerRowA + off] = tmp.z;
            sh_A[(innerColA * 4 + 3) * LDA + innerRowA + off] = tmp.w;
        }
        for (uint off = 0; off < BK; off += strideB)
        {
            *reinterpret_cast<float4*>(&sh_B[(innerRowB + off) * BN + innerColB * 4]) =
                *reinterpret_cast<const float4*>(&B[(innerRowB + off) * N + innerColB * 4]);
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dot = 0; dot < BK; dot++)
        {
            for (uint ii = 0; ii < TM; ii += 4)
            {
                float4 va = *reinterpret_cast<const float4*>(&sh_A[dot * LDA + inner_row * TM + ii]);
                reg_A[ii + 0] = va.x; reg_A[ii + 1] = va.y;
                reg_A[ii + 2] = va.z; reg_A[ii + 3] = va.w;
            }
            for (uint ii = 0; ii < TN; ii += 4)
            {
                float4 vb = *reinterpret_cast<const float4*>(&sh_B[dot * BN + inner_col * TN + ii]);
                reg_B[ii + 0] = vb.x; reg_B[ii + 1] = vb.y;
                reg_B[ii + 2] = vb.z; reg_B[ii + 3] = vb.w;
            }
            for (uint rr = 0; rr < TM; rr++)
                for (uint cc = 0; cc < TN; cc++)
                    value[rr * TN + cc] += reg_A[rr] * reg_B[cc];
        }
        __syncthreads();
    }

    for (uint rr = 0; rr < TM; rr++)
        for (uint cc = 0; cc < TN; cc += 4)
        {
            float4 v;
            v.x = value[rr * TN + cc + 0];
            v.y = value[rr * TN + cc + 1];
            v.z = value[rr * TN + cc + 2];
            v.w = value[rr * TN + cc + 3];
            *reinterpret_cast<float4*>(&C[(inner_row * TM + rr) * N + inner_col * TN + cc]) = v;
        }
}

#ifndef __CUDACC_RTC__
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
    thread_tiled_matmul<<<grid, block>>>(M, N, K, A, B, C);
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    const int iters = 50;
    cudaEventRecord(s);
    for (int it = 0; it < iters; it++) thread_tiled_matmul<<<grid, block>>>(M, N, K, A, B, C);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e); ms /= iters;
    printf("bank_conflict (padded)  BM=%d BN=%d BK=%d TM=%d TN=%d : %.3f ms, %.1f GFLOP/s\n",
           BM, BN, BK, TM, TN, ms, (2.0 * M * N * K) / (ms * 1e-3) / 1e9);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
#endif
