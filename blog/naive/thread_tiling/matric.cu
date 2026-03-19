#include <cstdio>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CU_CHECK(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* str = nullptr; \
        cuGetErrorString(err, &str); \
        std::printf("CU error at %s:%d - %s\n", __FILE__, __LINE__, str ? str : "unknown"); \
        return 1; \
    } \
} while (0)

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

int main()
{
    const int N_MIN = 4;
    const int N_MAX = 4097;

    const int BT = 16;
    const int TK = 32;
    const int WTX = 16;
    const int WTY = 2;
    const int TTX = 1;
    const int TTY = 1;

    if (WTX * WTY != 32) {
        std::printf("WTX*WTY must equal 32. Got %d*%d=%d\n", WTX, WTY, WTX * WTY);
        return 1;
    }
    if (BT % TTX != 0 || BT % TTY != 0) {
        std::printf("BT must be divisible by TTX and TTY. BT=%d TTX=%d TTY=%d\n", BT, TTX, TTY);
        return 1;
    }

    const int thread_cols = BT / TTX;
    const int thread_rows = BT / TTY;
    if (thread_cols % WTX != 0 || thread_rows % WTY != 0) {
        std::printf("(BT/TTX) must be divisible by WTX and (BT/TTY) by WTY.\n");
        std::printf("thread_cols=%d thread_rows=%d WTX=%d WTY=%d\n", thread_cols, thread_rows, WTX, WTY);
        return 1;
    }

    const int warp_tiles_per_row = thread_cols / WTX;
    const int warp_tiles_per_col = thread_rows / WTY;
    const int total_warps = warp_tiles_per_row * warp_tiles_per_col;
    const int threads = total_warps * 32;
    if (threads <= 0 || threads > 1024) {
        std::printf("Too many threads per block: %d (max 1024)\n", threads);
        return 1;
    }

    const size_t max_elems = (size_t)N_MAX * (size_t)N_MAX;
    const size_t max_bytes = max_elems * sizeof(float);

    std::vector<float> hA(max_elems);
    std::vector<float> hB(max_elems);

    for (size_t i = 0; i < max_elems; ++i) {
        hA[i] = sinf((float)i);
        hB[i] = cosf((float)i);
    }

    CU_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&context, device));
    CU_CHECK(cuCtxSetCurrent(context));

    float *A = nullptr, *B = nullptr, *C = nullptr;
    CUDA_CHECK(cudaMalloc(&A, max_bytes));
    CUDA_CHECK(cudaMalloc(&B, max_bytes));
    CUDA_CHECK(cudaMalloc(&C, max_bytes));

    CUmodule module;
    CUfunction kernel;
    CU_CHECK(cuModuleLoad(&module, "ptx.ptx"));
    CU_CHECK(cuModuleGetFunction(&kernel, module, "thread_tiled_ptx"));

    CUdeviceptr dA = (CUdeviceptr)A;
    CUdeviceptr dB = (CUdeviceptr)B;
    CUdeviceptr dC = (CUdeviceptr)C;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const size_t sharedMem = (size_t)(BT * TK + TK * BT) * sizeof(float);
    dim3 block(threads, 1, 1);

    std::printf("N,ms,gflops\n");

    for (int n = N_MIN; n <= N_MAX; n=n*2) {
        const size_t elems = (size_t)n * (size_t)n;
        const size_t bytes = elems * sizeof(float);

        CUDA_CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(C, 0, bytes));

        dim3 grid((n + BT - 1) / BT, (n + BT - 1) / BT, 1);

        void *args[] = {
            &dC, &dA, &dB,
            &n, &n, &n,
            (void*)&BT, (void*)&BT, (void*)&TK,
            (void*)&WTX, (void*)&WTY,
            (void*)&TTX, (void*)&TTY
        };

        CU_CHECK(cuLaunchKernel(kernel,
            grid.x, grid.y, 1,
            block.x, 1, 1,
            sharedMem, 0, args, 0));
        CU_CHECK(cuCtxSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        CU_CHECK(cuLaunchKernel(kernel,
            grid.x, grid.y, 1,
            block.x, 1, 1,
            sharedMem, 0, args, 0));
        CU_CHECK(cuCtxSynchronize());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        double gflops = 0.0;
        if (ms > 0.0f) {
            gflops = (2.0 * (double)n * (double)n * (double)n) / ((double)ms * 1.0e6);
        }

        std::printf("%d,%.6f,%.6f\n", n, ms, gflops);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    CU_CHECK(cuModuleUnload(module));
    CU_CHECK(cuDevicePrimaryCtxRelease(device));
    return 0;
}
