#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

constexpr int BT = 16;
constexpr int TK = 16;
constexpr int WM = 2;
constexpr int WN = 16;
constexpr int WARP_SIZE = 32;

__global__ void warp_block_matmul_cuda(const float* A, const float* B, float* C, int N)
{
    __shared__ float shA[BT * TK];
    __shared__ float shB[TK * BT];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_tiles_per_row = BT / WN;
    const int warp_row = warp_id / warp_tiles_per_row;
    const int warp_col = warp_id % warp_tiles_per_row;

    const int inner_row = warp_row * WM + lane_id / WN;
    const int inner_col = warp_col * WN + lane_id % WN;

    const int global_row = blockIdx.y * BT + inner_row;
    const int global_col = blockIdx.x * BT + inner_col;

    float sum = 0.0f;

    for (int kb = 0; kb < N; kb += TK) {
        for (int idx = tid; idx < BT * TK; idx += blockDim.x) {
            const int r = idx / TK;
            const int c = idx % TK;
            const int gr = blockIdx.y * BT + r;
            const int gc = kb + c;
            shA[idx] = (gr < N && gc < N) ? A[gr * N + gc] : 0.0f;
        }

        for (int idx = tid; idx < TK * BT; idx += blockDim.x) {
            const int r = idx / BT;
            const int c = idx % BT;
            const int gr = kb + r;
            const int gc = blockIdx.x * BT + c;
            shB[idx] = (gr < N && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        __syncthreads();

        if (global_row < N && global_col < N) {
            #pragma unroll
            for (int k = 0; k < TK; ++k) {
                const float a = shA[inner_row * TK + k];
                const float b = shB[k * BT + inner_col];
                sum = fmaf(a, b, sum);
            }
        }

        __syncthreads();
    }

    if (global_row < N && global_col < N) {
        C[global_row * N + global_col] = sum;
    }
}

int main()
{
    const int N = 1024;
    const int warps_per_col = BT / WM;
    const int warps_per_row = BT / WN;
    const int threads = warps_per_col * warps_per_row * WARP_SIZE;


    float *A = nullptr, *B = nullptr, *C = nullptr;
    CUDA_CHECK(cudaMalloc(&A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, N * N * sizeof(float)));

    float* hA = new float[N * N];
    float* hB = new float[N * N];
    float* hC = new float[N * N];

    for (int i = 0; i < N * N; ++i) hA[i] = sinf((float)i);
    for (int i = 0; i < N * N; ++i) hB[i] = cosf((float)i);

    CUDA_CHECK(cudaMemcpy(A, hA, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(C, 0, N * N * sizeof(float)));

    dim3 block(threads, 1, 1);
    dim3 grid((N + BT - 1) / BT, (N + BT - 1) / BT, 1);

    warp_block_matmul_cuda<<<grid, block>>>(A, B, C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int rep = 0; rep < 10; ++rep) {
        warp_block_matmul_cuda<<<grid, block>>>(A, B, C, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float ms = total_ms / 10.0f;

    CUDA_CHECK(cudaMemcpy(hC, C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    double max_error = 0.0;
    for (int row = 0; row < 64; ++row) {
        for (int col = 0; col < 64; ++col) {
            double cpu = 0.0;
            for (int k = 0; k < N; ++k) {
                cpu += (double)hA[row * N + k] * (double)hB[k * N + col];
            }
            double diff = std::fabs(cpu - (double)hC[row * N + col]);
            if (diff > max_error) max_error = diff;
            if (diff > 1e-2) {
                std::printf("Mismatch at (%d,%d): CPU=%f GPU=%f diff=%f\n", row, col, cpu, hC[row * N + col], diff);
                return 1;
            }
        }
    }

    const double gflops = (2.0 * (double)N * (double)N * (double)N) / ((double)ms * 1e6);
  
    std::printf("%f\n", gflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    delete[] hA;
    delete[] hB;
    delete[] hC;
    return 0;
}
