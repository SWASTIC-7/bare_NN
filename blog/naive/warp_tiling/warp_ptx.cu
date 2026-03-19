#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CU_CHECK(call) do { \
    CUresult err = call; \
    if(err != CUDA_SUCCESS) { \
        const char* str; \
        cuGetErrorString(err, &str); \
        printf("CU error at %s:%d - %s\n", __FILE__, __LINE__, str); \
        exit(1); \
    } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main()
{
    const int N = 1024;
    const int BT = 16;
    const int TK = 16;
    const int WM = 4;
    const int WN = 8;

    CU_CHECK(cuInit(0));
    CUdevice device; CUcontext context;
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&context, device));
    CU_CHECK(cuCtxSetCurrent(context));

    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, N*N*sizeof(float)));

    float *hA = new float[N*N];
    float *hB = new float[N*N];
    float *hC = new float[N*N];

    for(int i = 0; i < N*N; i++) hA[i] = sinf((float)i);
    for(int i = 0; i < N*N; i++) hB[i] = cosf((float)i);
    for(int i = 0; i < N*N; i++) hC[i] = 0.0f;

    CUDA_CHECK(cudaMemcpy(A, hA, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C, hC, N*N*sizeof(float), cudaMemcpyHostToDevice));

    CUmodule module; CUfunction kernel;
    CU_CHECK(cuModuleLoad(&module, "ptx.ptx"));
    CU_CHECK(cuModuleGetFunction(&kernel, module, "warp_tiled_ptx"));

    CUdeviceptr dA = (CUdeviceptr)A;
    CUdeviceptr dB = (CUdeviceptr)B;
    CUdeviceptr dC = (CUdeviceptr)C;

    if (WM * WN != 32) {
        printf("WM*WN must equal 32 (one warp). Got %d*%d=%d\n", WM, WN, WM * WN);
        return 1;
    }

    if (BT % WM != 0 || BT % WN != 0) {
        printf("BT must be divisible by WM and WN. BT=%d WM=%d WN=%d\n", BT, WM, WN);
        return 1;
    }

    int warps_per_col = BT / WM;
    int warps_per_row = BT / WN;
    int total_warps = warps_per_col * warps_per_row;
    int threads = total_warps * 32;

    if (threads > 1024) {
        printf("Too many threads per block: %d (max 1024)\n", threads);
        return 1;
    }



    size_t sharedMem = (BT * TK + TK * BT) * sizeof(float);
    dim3 block(threads, 1, 1);
    dim3 grid((N + BT - 1) / BT, (N + BT - 1) / BT, 1);

    void *args[] = {
        &dC, &dA, &dB,
        (void*)&N, (void*)&N, (void*)&N,
        (void*)&BT, (void*)&BT, (void*)&TK,
        (void*)&WN, (void*)&WM
    };

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemset(C, 0, N * N * sizeof(float)));

    // warmup
    CU_CHECK(cuLaunchKernel(kernel,
        grid.x, grid.y, 1,
        block.x, 1, 1,
        sharedMem, 0, args, 0));
    CU_CHECK(cuCtxSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int rep = 0; rep < 5; ++rep) {
        CU_CHECK(cuLaunchKernel(kernel,
            grid.x, grid.y, 1,
            block.x, 1, 1,
            sharedMem, 0, args, 0));
    }
    CU_CHECK(cuCtxSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float ms = total_ms / 5.0f;

    CUDA_CHECK(cudaMemcpy(hC, C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    double max_error = 0.0;
    for (int row = 0; row < 64; ++row) {
        for (int col = 0; col < 64; ++col) {
            double cpu = 0.0;
            for (int kk = 0; kk < N; ++kk) {
                cpu += (double)hA[row * N + kk] * (double)hB[kk * N + col];
            }
            double diff = fabs(cpu - (double)hC[row * N + col]);
            if (diff > max_error) max_error = diff;
            if (diff > 1e-2) {
                printf("Mismatch at (%d,%d): CPU=%f GPU=%f diff=%f\n",
                       row, col, cpu, hC[row * N + col], diff);
                return 1;
            }
        }
    }

    double gflops = (2.0 * (double)N * (double)N * (double)N) / ((double)ms * 1e6);
    
    printf("%f\n", gflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    delete[] hA; delete[] hB; delete[] hC;
    CU_CHECK(cuModuleUnload(module));
    CU_CHECK(cuDevicePrimaryCtxRelease(device));
    return 0;
}