# =============================================================================
# Common CUDA Utilities for bare_NN
# =============================================================================
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


#define CU_CHECK(call)                                                      \
    do {                                                                    \
        CUresult err = call;                                                \
        if (err != CUDA_SUCCESS) {                                          \
            const char* msg;                                                \
            cuGetErrorString(err, &msg);                                    \
            fprintf(stderr, "CUDA Driver Error %s:%d: %s\n",                \
                    __FILE__, __LINE__, msg);                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)


#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Runtime Error %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel launch check
#define KERNEL_CHECK()                                                      \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                               \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "Kernel Launch Error %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)


inline int calcGridSize(int n, int blockSize) {
    return (n + blockSize - 1) / blockSize;
}

// Calculate grid dimensions for 2D kernel launch
inline dim3 calcGridSize2D(int width, int height, dim3 blockSize) {
    return dim3(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
}

struct CudaTimer {
    cudaEvent_t start, stop;
    
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void begin() {
        cudaEventRecord(start);
    }
    
    float end() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

inline void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n", 
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared memory per block: %zu KB\n", 
               prop.sharedMemPerBlock / 1024);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Max grid size: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }
}
