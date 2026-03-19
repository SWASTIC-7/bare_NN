#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
int main() {

    int M = 1024;
    int N = 1024;
    int K = 1024;

    cuInit(0);

    CUdevice device;
    CUcontext context;

    cuDeviceGet(&device,0);
    cuDevicePrimaryCtxRetain(&context, device);
    cuCtxSetCurrent(context);

    float *A,*B,*C;

    cudaMalloc(&A,M*K*sizeof(float));
    cudaMalloc(&B,K*N*sizeof(float));
    cudaMalloc(&C,M*N*sizeof(float));

    float *hA = new float[M*K];
    float *hB = new float[K*N];
    float *hC = new float[M*N];

    // Initialize matrices
    for(int i=0;i<M*K;i++)
        hA[i] = sin(i);

    for(int i=0;i<K*N;i++)
        hB[i] = cos(i);

    for(int i=0;i<M*N;i++)
        hC[i] = 0.0f;

    cudaMemcpy(A,hA,M*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B,hB,K*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(C,hC,M*N*sizeof(float),cudaMemcpyHostToDevice);

    // Load PTX
    CUmodule module;
    CUfunction kernel;

    cuModuleLoad(&module,"ptx.ptx");
    cuModuleGetFunction(&kernel,module,"naive_ptx");

    dim3 block(32,32);
    dim3 grid((N+31)/32,(M+31)/32);

    void *args[] = {
        &C,
        &A,
        &B,
        &M,
        &N,
        &K
    };
    //warm-up
        cuLaunchKernel(
        kernel,
        grid.x,grid.y,1,
        block.x,block.y,1,
        0,
        0,
        args,
        0
    );

    cudaDeviceSynchronize();

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    cudaEventRecord(start);
    cuLaunchKernel(
        kernel,
        grid.x,grid.y,1,
        block.x,block.y,1,
        0,
        0,
        args,
        0
    );

    // IMPORTANT: wait for driver kernel
    // cuCtxSynchronize();


//     cudaMemcpy(hC,C,M*N*sizeof(float),cudaMemcpyDeviceToHost);
//     double checksum = 0;
// double max_error = 0;

// for(int row = 0; row < M; row++)
// {
//     for(int col = 0; col < N; col++)
//     {
//         double cpu_sum = 0.0;

//         for(int k = 0; k < K; k++)
//         {
//             cpu_sum += hA[row*K + k] * hB[k*N + col];
//         }

//         double gpu_val = hC[row*N + col];
//         double diff = fabs(cpu_sum - gpu_val);

//         if(diff > max_error)
//             max_error = diff;

//         // assert equality up to 3 decimal places
//         if(diff > 1e-3)
//         {
//             printf("Mismatch at (%d,%d): CPU=%f GPU=%f diff=%f\n",
//                    row, col, cpu_sum, gpu_val, diff);
//             exit(1);
//         }

//         checksum += gpu_val;
//     }
// }

// std::cout << "checksum " << checksum << std::endl;
// std::cout << "max error " << max_error << std::endl;
// std::cout << "Validation passed!" << std::endl;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    double gflops = (2.0 * M * N * K) / (ms * 1e6);

    std::cout <<  gflops << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    delete[] hA;
    delete[] hB;
    delete[] hC;

    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(device);

    return 0;
}