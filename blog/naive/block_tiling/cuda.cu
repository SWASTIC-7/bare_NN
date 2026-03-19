
// __global__ void tiled_thread_matmul(float *A, float *B, float *C, int N)
// {

//     const uint block_row = blockIdx.y;
//     const uint block_col = blockIdx.x;

//     const uint inner_row = threadId.x / TILE_WIDTH;
//     const uint inner_col = threadId.x % TILE_WIDTH;

//     int row = block_row * TILE_WIDTH + inner_row;
//     int col = block_col * TILE_WIDTH + inner_col;

//     __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
//     __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

//     float value = 0;

//     for (int i = 0; i < N / TILE_WIDTH; i++)
//     {
//         sh_A[inner_row][inner_col] = A[row * N + i * TILE_WIDTH + inner_col];
//         sh_B[inner_row][inner_col] = B[(i * TILE_WIDTH + inner_row) * N + col];

//         __syncthreads();

//         for (int k = 0; k < TILE_WIDTH; k++)
//             value += sh_A[inner_row][k] * sh_B[k][inner_col];   
//         __syncthreads();
//     }

//     C[row * N + col] = value;
// }


// #define TILE_M 64
// #define TILE_N 64
// #define TILE_K 16

// __global__ void tiled_matmul(float *A, float *B, float *C, int N)
// {
//     const int block_row = blockIdx.y;
//     const int block_col = blockIdx.x;

//     const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

//     // thread coordinates inside tile
//     const int inner_row = thread_id / TILE_N;
//     const int inner_col = thread_id % TILE_N;

//     int row = block_row * TILE_M + inner_row;
//     int col = block_col * TILE_N + inner_col;

//     __shared__ float sh_A[TILE_M][TILE_K];
//     __shared__ float sh_B[TILE_K][TILE_N];

//     float value = 0.0f;

//     for (int tile = 0; tile < N; tile += TILE_K)
//     {
//         // load A tile
//         if(inner_row < TILE_M && inner_col < TILE_K)
//         {
//             sh_A[inner_row][inner_col] =
//                 A[row * N + tile + inner_col];
//         }

//         // load B tile
//         if(inner_row < TILE_K && inner_col < TILE_N)
//         {
//             sh_B[inner_row][inner_col] =
//                 B[(tile + inner_row) * N + col];
//         }

//         __syncthreads();

//         // compute partial product
//         for (int k = 0; k < TILE_K; k++)
//         {
//             value += sh_A[inner_row][k] * sh_B[k][inner_col];
//         }

//         __syncthreads();
//     }

//     C[row * N + col] = value;
// }
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__global__ void tiled_cuda(
    float* C, const float* A, const float* B,
    int M, int N, int K,
    int BT_M, int BT_N, int BT_K)
{
    extern __shared__ float smem[];
    float* sh_A = smem;
    float* sh_B = smem + BT_M * BT_K;

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int tidx      = threadIdx.x;

    const int inner_row_C = tidx / BT_N;
    const int inner_col_C = tidx % BT_N;
    const int inner_row_A = tidx / BT_K;
    const int inner_col_A = tidx % BT_K;
    const int inner_row_B = tidx / BT_N;
    const int inner_col_B = tidx % BT_N;

    const int row = block_row * BT_M + inner_row_C;
    const int col = block_col * BT_N + inner_col_C;

    float sum = 0.0f;

    for (int i = 0; i < K; i += BT_K) {

        // Load A tile
        if (inner_row_A < BT_M && inner_col_A < BT_K) {
            int grow = block_row * BT_M + inner_row_A;
            int gcol = i + inner_col_A;
            sh_A[inner_row_A * BT_K + inner_col_A] =
                (grow < M && gcol < K) ? A[grow * K + gcol] : 0.0f;
        }

        // Load B tile
        if (inner_row_B < BT_K && inner_col_B < BT_N) {
            int grow = i + inner_row_B;
            int gcol = block_col * BT_N + inner_col_B;
            sh_B[inner_row_B * BT_N + inner_col_B] =
                (grow < K && gcol < N) ? B[grow * N + gcol] : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BT_K; k++)
            sum += sh_A[inner_row_C * BT_K + k] *
                   sh_B[k * BT_N + inner_col_C];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const int BT_M = 32, BT_N = 32, BT_K = 32;

    // --- host alloc and init ---
    float* hA = new float[M * K];
    float* hB = new float[K * N];
    float* hC = new float[M * N];

    for (int i = 0; i < M * K; i++) hA[i] = sinf((float)i);
    for (int i = 0; i < K * N; i++) hB[i] = cosf((float)i);
    for (int i = 0; i < M * N; i++) hC[i] = 0.0f;

    // --- device alloc ---
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));

    cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BT_M * BT_N);
    dim3 grid((N + BT_N - 1) / BT_N, (M + BT_M - 1) / BT_M);
    size_t smem_size = (BT_M * BT_K + BT_K * BT_N) * sizeof(float);

    // --- warmup ---
    tiled_cuda<<<grid, block, smem_size>>>(dC, dA, dB, M, N, K, BT_M, BT_N, BT_K);
    cudaDeviceSynchronize();

    // --- timed run ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tiled_cuda<<<grid, block, smem_size>>>(dC, dA, dB, M, N, K, BT_M, BT_N, BT_K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    printf("%f\n", gflops);

    // --- copy back and validate ---
    cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);


done:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}