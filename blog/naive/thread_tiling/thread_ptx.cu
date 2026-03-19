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
    const int TK = 32;
    const int WTX = 16;
    const int WTY = 2;
    const int TTX = 1;
    const int TTY = 1;

    CU_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&context, device));
    CU_CHECK(cuCtxSetCurrent(context));

    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, N * N * sizeof(float)));

    float *hA = new float[N * N];
    float *hB = new float[N * N];
    float *hC = new float[N * N];

    for (int i = 0; i < N * N; i++) hA[i] = sinf((float)i);
    for (int i = 0; i < N * N; i++) hB[i] = cosf((float)i);
    for (int i = 0; i < N * N; i++) hC[i] = 0.0f;

    CUDA_CHECK(cudaMemcpy(A, hA, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C, hC, N * N * sizeof(float), cudaMemcpyHostToDevice));

    CUmodule module;
    CUfunction kernel;
    CU_CHECK(cuModuleLoad(&module, "ptx.ptx"));
    CU_CHECK(cuModuleGetFunction(&kernel, module, "thread_tiled_ptx"));

    CUdeviceptr dA = (CUdeviceptr)A;
    CUdeviceptr dB = (CUdeviceptr)B;
    CUdeviceptr dC = (CUdeviceptr)C;

    if (WTX * WTY != 32) {
        printf("WTX*WTY must equal 32. Got %d*%d=%d\n", WTX, WTY, WTX * WTY);
        return 1;
    }
    if (BT % TTX != 0 || BT % TTY != 0) {
        printf("BT must be divisible by TTX and TTY. BT=%d TTX=%d TTY=%d\n", BT, TTX, TTY);
        return 1;
    }

    const int thread_cols = BT / TTX;
    const int thread_rows = BT / TTY;
    if (thread_cols % WTX != 0 || thread_rows % WTY != 0) {
        printf("(BT/TTX) must be divisible by WTX and (BT/TTY) by WTY.\n");
        printf("thread_cols=%d thread_rows=%d WTX=%d WTY=%d\n", thread_cols, thread_rows, WTX, WTY);
        return 1;
    }

    const int warp_tiles_per_row = thread_cols / WTX;
    const int warp_tiles_per_col = thread_rows / WTY;
    const int total_warps = warp_tiles_per_row * warp_tiles_per_col;
    const int threads = total_warps * 32;
    if (threads <= 0 || threads > 1024) {
        printf("Too many threads per block: %d (max 1024)\n", threads);
        return 1;
    }

    printf("Config: BT=%d TK=%d WTX=%d WTY=%d TTX=%d TTY=%d\n", BT, TK, WTX, WTY, TTX, TTY);
    printf("Threads: %d (warps %d x %d = %d)\n", threads, warp_tiles_per_col, warp_tiles_per_row, total_warps);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const size_t sharedMem = (BT * TK + TK * BT) * sizeof(float);
    dim3 block(threads, 1, 1);
    dim3 grid((N + BT - 1) / BT, (N + BT - 1) / BT, 1);

    void *args[] = {
        &dC, &dA, &dB,
        (void*)&N, (void*)&N, (void*)&N,
        (void*)&BT, (void*)&BT, (void*)&TK,
        (void*)&WTX, (void*)&WTY,
        (void*)&TTX, (void*)&TTY
    };

    CUDA_CHECK(cudaMemset(C, 0, N * N * sizeof(float)));

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
    const float ms = total_ms / 5.0f;

    CUDA_CHECK(cudaMemcpy(hC, C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (int idx = 0; idx < N * N; ++idx) checksum += (double)hC[idx];

    double max_error = 0.0;
    int error_count = 0;
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            double cpu = 0.0;
            for (int kk = 0; kk < N; ++kk) {
                cpu += (double)hA[row * N + kk] * (double)hB[kk * N + col];
            }
            const double diff = fabs(cpu - (double)hC[row * N + col]);
            if (diff > max_error) max_error = diff;
            if (diff > 1e-2) error_count++;
        }
    }

    const double gflops = (2.0 * (double)N * (double)N * (double)N) / ((double)ms * 1e6);
    printf("Checksum:     %.10e\n", checksum);
    printf("Max error:    %.10e\n", max_error);
    printf("Error count:  %d\n", error_count);
    printf("Time:         %.3f ms\n", ms);
    printf("Performance:  %.2f GFLOPS\n", gflops);

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
