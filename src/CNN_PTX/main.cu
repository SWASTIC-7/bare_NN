
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"
#include <cuda.h>
#include <cstdio>
#include <time.h>

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

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main() {

    CHECK(cuInit(0));

    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));

    CUcontext ctx;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));
    
    loaddata();

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