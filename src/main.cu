#include <cuda.h>
#include <stdio.h>

// Error check macro
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

    CUmodule module;
    CHECK(cuModuleLoad(&module, "ptx/hello.ptx"));

    CUfunction kernel;
    CHECK(cuModuleGetFunction(&kernel, module, "hello"));

    unsigned int host_out = 0;
    CUdeviceptr dev_out;
    CHECK(cuMemAlloc_v2(&dev_out, sizeof(unsigned int))); 

    void* args[] = {
        &dev_out
    };

    CHECK(cuLaunchKernel(
        kernel,
        1, 1, 1,    // grid dims
        1, 1, 1,    // block dims
        0,          // shared memory
        0,          // stream
        args,
        nullptr
    ));

    CHECK(cuCtxSynchronize());

    CHECK(cuMemcpyDtoH_v2(&host_out, dev_out, sizeof(unsigned int)));

    printf("GPU says: %u\n", host_out);


    cuMemFree_v2(dev_out);
    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(dev);

    return 0;
}
