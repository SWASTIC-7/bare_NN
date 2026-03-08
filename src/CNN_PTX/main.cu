
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


static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main() {

    loaddata();

    // CHECK(cuInit(0));

    // CUdevice dev;
    // CHECK(cuDeviceGet(&dev, 0));

    // CUcontext ctx;
    // CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    // CHECK(cuCtxSetCurrent(ctx));
    
    // loaddata();

    // CUmodule module;
    // CHECK(cuModuleLoad(&module, "ptx/hello.ptx"));

    // CUfunction kernel;
    // CHECK(cuModuleGetFunction(&kernel, module, "hello"));

    // unsigned int host_out = 0;
    // CUdeviceptr dev_out;
    // CHECK(cuMemAlloc_v2(&dev_out, sizeof(unsigned int))); 

    // void* args[] = {
    //     &dev_out
    // };

    // CHECK(cuLaunchKernel(
    //     kernel,
    //     1, 1, 1,    // grid dims
    //     1, 1, 1,    // block dims
    //     0,          // shared memory
    //     0,          // stream
    //     args,
    //     nullptr
    // ));

    // CHECK(cuCtxSynchronize());

    // CHECK(cuMemcpyDtoH_v2(&host_out, dev_out, sizeof(unsigned int)));

    // printf("GPU says: %u\n", host_out);


    // cuMemFree_v2(dev_out);
    // cuModuleUnload(module);
    // cuDevicePrimaryCtxRelease(dev);

    return 0;
}

static void learn() {

    // Initialize cuBLAS
    static cublasHandle_t blas;
	cublasCreate(&blas);

    fprintf(stdout ,"Starting Learning Process\n");

    float error;

    int max_iter = 50;

    double time_taken = 0.0;
    double total_time = 0.0;
    // epoch loop
    while (max_iter < 0 || max_iter-- > 0) {

        error = 0.0f;

        for (int i = 0; i < train_cnt; ++i) {
			float tmp_err;

			time_taken += forward_pass(train_set[i].data);

			l_f.bp_clear();
			l_s1.bp_clear();
			l_c1.bp_clear();

			// Euclid distance of train_set[i]
            // TODO: replace this with ptx code for makeError
			makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
            // calculate norm2 usign cublas
            // TODO: replace this with ptx code for norm2
			cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp_err);
			err += tmp_err;

			time_taken += back_pass();
		}

        error /= train_cnt;
		fprintf(stdout, "error: %e, time taken for epoch: %i is %lf\n", error, i, time_taken);
        total_time += time_taken;
        time_taken = 0.0;
		if (error < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}
    }

    fprintf(stdout, "\n Total Time Taken - %lf\n", total_time);
}

// Unfold the input layer
// 5*5 kernel, stride 1, no padding, so output is 24*24*5*5
static void unfold_input(double input[28][28], double unfolded[24*24][5*5])
{
	int a = 0;
	(void)unfold_input;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j) {
			int b = 0;
			for (int x = i; x < i + 2; ++x)
				for (int y = j; y < j+2; ++y)
					unfolded[a][b++] = input[x][y];
			a++;
		}
}