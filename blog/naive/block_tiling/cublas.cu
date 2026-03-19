#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define M 1024
#define N 1024
#define K 1024

int main()
{
    float *A, *B, *C;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate device memory (NOT unified memory)
    cudaMalloc(&A, sizeA);
    cudaMalloc(&B, sizeB);
    cudaMalloc(&C, sizeC);

    // Host memory
    float *hA = new float[M*K];
    float *hB = new float[K*N];
    float *hC = new float[M*N];

    for (int i = 0; i < M * K; i++)
        hA[i] = sin(i);

    for (int i = 0; i < K * N; i++)
        hB[i] = cos(i);

    for (int i = 0; i < M * N; i++)
        hC[i] = 0.0f;

    cudaMemcpy(A, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(C, hC, sizeC, cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Warm-up run (important)
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double gflops = (2.0 * M * N * K) / (ms * 1e6);

    std::cout <<  gflops << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    delete[] hA;
    delete[] hB;
    delete[] hC;

    cublasDestroy(handle);

    return 0;
}