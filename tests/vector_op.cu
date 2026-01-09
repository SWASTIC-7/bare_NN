#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Error checking macro
#define CHECK(call)                                                     \
    do {                                                                \
        CUresult err = call;                                            \
        if (err != CUDA_SUCCESS) {                                      \
            const char* msg;                                            \
            cuGetErrorString(err, &msg);                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                    __FILE__, __LINE__, msg);                           \
            return 1;                                                   \
        }                                                               \
    } while (0)

int main() {

    CHECK(cuInit(0));

    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));

    CUcontext ctx;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));

    int N = 1024;
    int BLOCK_SIZE = 256;
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocating host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];
    
    // Initialize test data
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i + 1);
        h_b[i] = static_cast<float>(N - i + 1);
    }

    CUmodule module;
    CHECK(cuModuleLoad(&module, "ptx/vector_operation.ptx"));

    CUfunction kernel;
    CHECK(cuModuleGetFunction(&kernel, module, "vectorScalarMul"));

   
    CUdeviceptr dev_a, dev_b, dev_c;
    CHECK(cuMemAlloc_v2(&dev_a, N * sizeof(float)));
    CHECK(cuMemAlloc_v2(&dev_b, N * sizeof(float)));
    CHECK(cuMemAlloc_v2(&dev_c, N * sizeof(float)));

    
    CHECK(cuMemcpyHtoD_v2(dev_a, h_a, N * sizeof(float)));
    CHECK(cuMemcpyHtoD_v2(dev_b, h_b, N * sizeof(float)));

    float scalar = 2.5f;  
    
    void* args[] = {
        &dev_a,
        &scalar,      
        &dev_c,
        &N
    };

    CHECK(cuLaunchKernel(
        kernel,
        GRID_SIZE, 1, 1,    // grid dims
        BLOCK_SIZE, 1, 1,    // block dims
        0,          // shared memory
        0,          // stream
        args,
        nullptr
    ));

    CHECK(cuCtxSynchronize());

    CHECK(cuMemcpyDtoH_v2(h_c, dev_c, N * sizeof(float)));

    printf("multiplying %f * %f = %f (expected: %f)\n", h_a[0], scalar, h_c[0], h_a[0] * scalar);


    cuMemFree_v2(dev_a);
    cuMemFree_v2(dev_b);
    cuMemFree_v2(dev_c);
    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(dev);

    return 0;
}
