#include <iostream>
#include <cuda_runtime.h>
#include <cmath>


#define M 1024
#define N 1024
#define K 1024

__global__ void matmul_naive(
    int m, int n, int k,
    const float *A,
    const float *B,
    float *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        float sum = 0.0f;

        for (int i = 0; i < k; i++)
        {
            sum += A[row * k + i] * B[i * n + col];
        }

        C[row * n + col] = sum;
    }
}

int main()
{
    float *A, *B, *C;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    cudaMallocManaged(&A, sizeA);
    cudaMallocManaged(&B, sizeB);
    cudaMallocManaged(&C, sizeC);

    for (int i = 0; i < M * K; i++)
        A[i] = sin(i);

    for (int i = 0; i < K * N; i++)
        B[i] = cos(i);

    for (int i = 0; i < M * N; i++)
        C[i] = 0.0f;

    dim3 block(32,32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    matmul_naive<<<grid, block>>>(M, N, K, A, B, C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double gflops = (2.0 * M * N * K) / (ms * 1e6);

    std::cout << gflops << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}