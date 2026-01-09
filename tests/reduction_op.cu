#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

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

    unsigned int N = 256;
    
    // Initialize test data: 1, 2, 3, ..., N
    float *h_a = (float*)malloc(N * sizeof(float));
    float expected_sum = 0.0f;
    for (unsigned int i = 0; i < N; i++) {
        h_a[i] = (float)(i + 1);
        expected_sum += h_a[i];
    }

    // Load PTX module
    CUmodule module;
    CHECK(cuModuleLoad(&module, "../ptx/reduction_op.ptx"));

    CUfunction kernel;
    CHECK(cuModuleGetFunction(&kernel, module, "sum"));

    // Allocate device memory
    CUdeviceptr dev_a, dev_sum;
    CHECK(cuMemAlloc_v2(&dev_a, N * sizeof(float)));
    CHECK(cuMemAlloc_v2(&dev_sum, sizeof(float)));
    CHECK(cuMemcpyHtoD_v2(dev_a, h_a, N * sizeof(float)));
    
    void* args[] = { &dev_a, &dev_sum, &N };

    CHECK(cuLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, 0, args, nullptr));
    CHECK(cuCtxSynchronize());

    float sum_host = 0.0f;
    CHECK(cuMemcpyDtoH_v2(&sum_host, dev_sum, sizeof(float)));

    printf("Sum: %f (expected: %f) - %s\n", sum_host, expected_sum,
           sum_host == expected_sum ? "PASS" : "FAIL");

    // Cleanup
    free(h_a);
    cuMemFree_v2(dev_a);
    cuMemFree_v2(dev_sum);
    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(dev);

    return 0;
}
