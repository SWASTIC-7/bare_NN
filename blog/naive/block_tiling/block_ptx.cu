#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CU_CHECK(call) do { \
    CUresult err = call; \
    if(err != CUDA_SUCCESS) { \
        const char* str; \
        cuGetErrorString(err, &str); \
        printf("CU error at %s:%d — %s\n", __FILE__, __LINE__, str); \
        exit(1); \
    } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main() {

    int M = 1024, N = 1024, K = 1024;

    CU_CHECK(cuInit(0));

    CUdevice device;
    CUcontext context;
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&context, device));
    CU_CHECK(cuCtxSetCurrent(context));

    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, K*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, M*N*sizeof(float)));

    float *hA = new float[M*K];
    float *hB = new float[K*N];
    float *hC = new float[M*N];

    for(int i = 0; i < M*K; i++) hA[i] = sinf(i);
    for(int i = 0; i < K*N; i++) hB[i] = cosf(i);
    for(int i = 0; i < M*N; i++) hC[i] = 0.0f;

    CUDA_CHECK(cudaMemcpy(A, hA, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB, K*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C, hC, M*N*sizeof(float), cudaMemcpyHostToDevice));

    CUmodule module;
    CUfunction kernel;
    CU_CHECK(cuModuleLoad(&module, "ptx.ptx"));
    CU_CHECK(cuModuleGetFunction(&kernel, module, "tiled_ptx"));

    int BT_M = 32, BT_N = 32, BT_K = 32;

    dim3 block(BT_N, BT_M);
    dim3 grid((N + BT_N - 1) / BT_N,
              (M + BT_M - 1) / BT_M);

    CUdeviceptr dA = (CUdeviceptr)A;
    CUdeviceptr dB = (CUdeviceptr)B;
    CUdeviceptr dC = (CUdeviceptr)C;

    void *args[] = { &dC, &dA, &dB, &M, &N, &K, &BT_M, &BT_N, &BT_K };

    size_t sharedMemBytes = (BT_M * BT_K + BT_K * BT_N) * sizeof(float);

    // warm-up
    CU_CHECK(cuLaunchKernel(kernel,
        grid.x, grid.y, 1,
        block.x * block.y, 1, 1,
        sharedMemBytes, 0, args, 0));
    CU_CHECK(cuCtxSynchronize());

    // timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    CU_CHECK(cuLaunchKernel(kernel,
        grid.x, grid.y, 1,
        block.x * block.y, 1, 1,
        sharedMemBytes, 0, args, 0));

    CU_CHECK(cuCtxSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // copy back
    CUDA_CHECK(cudaMemcpy(hC, C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    // validate
    // double checksum = 0, max_error = 0;
    // for(int row = 0; row < M; row++) {
    //     for(int col = 0; col < N; col++) {
    //         double cpu_sum = 0.0;
    //         for(int k = 0; k < K; k++)
    //             cpu_sum += (double)hA[row*K + k] * (double)hB[k*N + col];

    //         double gpu_val = hC[row*N + col];
    //         double diff = fabs(cpu_sum - gpu_val);
    //         if(diff > max_error) max_error = diff;

    //         if(diff > 1e-3) {
    //             printf("Mismatch at (%d,%d): CPU=%f GPU=%f diff=%f\n",
    //                    row, col, cpu_sum, gpu_val, diff);
    //             return 1;
    //         }
    //         checksum += gpu_val;
    //     }
    // }

    double gflops = (2.0 * M * N * K) / (ms * 1e6);

  
    printf("%f\n", gflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));

    delete[] hA;
    delete[] hB;
    delete[] hC;

    CU_CHECK(cuModuleUnload(module));
    CU_CHECK(cuDevicePrimaryCtxRelease(device));

    return 0;
}