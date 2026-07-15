// Double-buffered vectorized CUDA thread-tiled matmul (register-staged prefetch),
// the CUDA mirror of buffering.ptx. Two shared buffers ping-pong; the next tile's
// global loads are issued into registers BEFORE the compute and stored to the other
// buffer AFTER it, so global-load latency overlaps the FMAs.
//   sh_A padded+transposed [2][BK][LDA] (LDA=BM+4) , sh_B [2][BK][BN]
// Requires each thread to load <=1 float4 of A and of B  (blockDim.x >= BM*BK/4 and >= BK*BN/4).
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
#define LDA (BM + 4)

extern "C" __global__ void thread_tiled_matmul(
        int M, int N, int K,
        const float *A, const float *B, float *C)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;
    const uint threads_per_row = BN / TN;
    const uint inner_row = threadIdx.x / threads_per_row;
    const uint inner_col = threadIdx.x % threads_per_row;

    __shared__ __align__(16) float sh_A[2][BK * LDA];   // transposed + padded
    __shared__ __align__(16) float sh_B[2][BK * BN];

    A += block_row * BM * K;
    B += block_col * BN;
    C += block_row * BM * N + block_col * BN;

    // this thread's single float4 coords for A and B, and whether it loads at all
    const uint aA_row  = threadIdx.x / (BK / 4);
    const uint aA_colk = (threadIdx.x % (BK / 4)) * 4;
    const bool aActive = threadIdx.x < (BM * BK / 4);
    const uint bB_row  = threadIdx.x / (BN / 4);
    const uint bB_coln = (threadIdx.x % (BN / 4)) * 4;
    const bool bActive = threadIdx.x < (BK * BN / 4);

    float value[TM * TN] = {0.0f};
    float reg_A[TM];
    float reg_B[TN];
    float4 pa, pb;                       // prefetched next tile, held in registers across compute

    // prologue: load tile @ bk=0 into buffer 0
    if (aActive)
    {
        float4 t = *reinterpret_cast<const float4*>(&A[aA_row * K + aA_colk]);
        sh_A[0][(aA_colk + 0) * LDA + aA_row] = t.x;
        sh_A[0][(aA_colk + 1) * LDA + aA_row] = t.y;
        sh_A[0][(aA_colk + 2) * LDA + aA_row] = t.z;
        sh_A[0][(aA_colk + 3) * LDA + aA_row] = t.w;
    }
    if (bActive)
        *reinterpret_cast<float4*>(&sh_B[0][bB_row * BN + bB_coln]) =
            *reinterpret_cast<const float4*>(&B[bB_row * N + bB_coln]);

    int cur = 0;
    for (uint bk = 0; bk < K; bk += BK)
    {
        __syncthreads();                 // buffer 'cur' is ready
        int nxt = cur ^ 1;
        uint next_bk = bk + BK;

        // prefetch-load next tile into registers (issue global loads, don't wait)
        if (next_bk < K)
        {
            if (aActive) pa = *reinterpret_cast<const float4*>(&A[aA_row * K + next_bk + aA_colk]);
            if (bActive) pb = *reinterpret_cast<const float4*>(&B[next_bk * N + bB_row * N + bB_coln]);
        }

        // compute current tile from buffer 'cur' (overlaps the prefetch loads)
        for (uint dot = 0; dot < BK; dot++)
        {
            for (uint ii = 0; ii < TM; ii += 4)
            {
                float4 va = *reinterpret_cast<const float4*>(&sh_A[cur][dot * LDA + inner_row * TM + ii]);
                reg_A[ii + 0] = va.x; reg_A[ii + 1] = va.y;
                reg_A[ii + 2] = va.z; reg_A[ii + 3] = va.w;
            }
            for (uint ii = 0; ii < TN; ii += 4)
            {
                float4 vb = *reinterpret_cast<const float4*>(&sh_B[cur][dot * BN + inner_col * TN + ii]);
                reg_B[ii + 0] = vb.x; reg_B[ii + 1] = vb.y;
                reg_B[ii + 2] = vb.z; reg_B[ii + 3] = vb.w;
            }
            for (uint rr = 0; rr < TM; rr++)
                for (uint cc = 0; cc < TN; cc++)
                    value[rr * TN + cc] += reg_A[rr] * reg_B[cc];
        }

        // prefetch-store registers into buffer 'nxt' (global data has arrived during compute)
        if (next_bk < K)
        {
            if (aActive)
            {
                sh_A[nxt][(aA_colk + 0) * LDA + aA_row] = pa.x;
                sh_A[nxt][(aA_colk + 1) * LDA + aA_row] = pa.y;
                sh_A[nxt][(aA_colk + 2) * LDA + aA_row] = pa.z;
                sh_A[nxt][(aA_colk + 3) * LDA + aA_row] = pa.w;
            }
            if (bActive) *reinterpret_cast<float4*>(&sh_B[nxt][bB_row * BN + bB_coln]) = pb;
        }
        cur = nxt;
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
    printf("double-buffered  BM=%d BN=%d BK=%d TM=%d TN=%d : %.3f ms, %.1f GFLOP/s\n",
           BM, BN, BK, TM, TN, ms, (2.0 * M * N * K) / (ms * 1e-3) / 1e9);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
#endif
