#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Error check macro for Driver API
#define CHECK(call)                                                     \
    do {                                                                \
        CUresult err = call;                                            \
        if (err != CUDA_SUCCESS) {                                      \
            const char* msg;                                            \
            cuGetErrorString(err, &msg);                                \
            fprintf(stderr, "CUDA Driver error %s:%d: %s\n",            \
                    __FILE__, __LINE__, msg);                           \
            exit(1);                                                    \
        }                                                               \
    } while (0)

// Error check macro for Runtime API
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA Runtime error %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(1);                                                    \
        }                                                               \
    } while (0)

// Complex number structure
struct Complex {
    float real;
    float imag;
};

// ============================================================================
// Standard CUDA Runtime Kernel for Complex Matrix Multiplication
// ============================================================================

__global__ void cuda_mul_matrix_complex(Complex* c, const Complex* a, const Complex* b,
                                        int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= M * N) return;
    
    int y = idx / N;  // row in output
    int x = idx % N;  // col in output
    
    Complex sum = {0.0f, 0.0f};
    
    for (int k = 0; k < K; k++) {
        Complex a_val = a[y * K + k];
        Complex b_val = b[k * N + x];
        
        // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        float real_part = a_val.real * b_val.real - a_val.imag * b_val.imag;
        float imag_part = a_val.real * b_val.imag + a_val.imag * b_val.real;
        
        sum.real += real_part;
        sum.imag += imag_part;
    }
    
    c[idx] = sum;
}

// ============================================================================
// Helper Functions
// ============================================================================

void print_complex_matrix(const char* name, Complex* matrix, int rows, int cols) {
    printf("\n%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows && i < 4; i++) {  // Print max 4 rows
        for (int j = 0; j < cols && j < 4; j++) {  // Print max 4 cols
            Complex val = matrix[i * cols + j];
            printf("(%.2f%+.2fi) ", val.real, val.imag);
        }
        if (cols > 4) printf("...");
        printf("\n");
    }
    if (rows > 4) printf("...\n");
}

bool compare_results(Complex* a, Complex* b, int size, float epsilon = 1e-3) {
    for (int i = 0; i < size; i++) {
        float diff_real = fabsf(a[i].real - b[i].real);
        float diff_imag = fabsf(a[i].imag - b[i].imag);
        if (diff_real > epsilon || diff_imag > epsilon) {
            printf("Mismatch at index %d: PTX(%.6f%+.6fi) vs CUDA(%.6f%+.6fi)\n",
                   i, a[i].real, a[i].imag, b[i].real, b[i].imag);
            return false;
        }
    }
    return true;
}

// ============================================================================
// Main Program
// ============================================================================

int main() {
    printf("Complex Matrix Multiplication: PTX vs CUDA Runtime Kernel\n");
    printf("=========================================================\n\n");
    
    // Matrix dimensions: C(M x N) = A(M x K) * B(K x N)
    int M = 128;
    int N = 128;
    int K = 128;
    
    printf("Matrix dimensions: C(%dx%d) = A(%dx%d) * B(%dx%d)\n", M, N, M, K, K, N);
    
    // Allocate host memory
    int size_a = M * K;
    int size_b = K * N;
    int size_c = M * N;
    
    Complex* h_a = new Complex[size_a];
    Complex* h_b = new Complex[size_b];
    Complex* h_c_ptx = new Complex[size_c];
    Complex* h_c_cuda = new Complex[size_c];
    
    // Initialize matrices with test data
    printf("Initializing matrices...\n");
    for (int i = 0; i < size_a; i++) {
        h_a[i].real = (float)(rand() % 100) / 10.0f;
        h_a[i].imag = (float)(rand() % 100) / 10.0f;
    }
    for (int i = 0; i < size_b; i++) {
        h_b[i].real = (float)(rand() % 100) / 10.0f;
        h_b[i].imag = (float)(rand() % 100) / 10.0f;
    }
    
    print_complex_matrix("Matrix A", h_a, M, K);
    print_complex_matrix("Matrix B", h_b, K, N);
    
    // ========================================================================
    // Method 1: PTX Kernel using CUDA Driver API
    // ========================================================================
    
    printf("\n--- Running PTX Kernel (Driver API) ---\n");
    
    // Initialize CUDA Driver API
    CHECK(cuInit(0));
    
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    
    CUcontext ctx;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));
    
    // Load PTX module
    CUmodule module;
    CHECK(cuModuleLoad(&module, "ptx/matmul.ptx"));
    
    // Get kernel function
    CUfunction kernel;
    CHECK(cuModuleGetFunction(&kernel, module, "ptx_mul_matrix_complex"));
    
    // Allocate device memory for PTX kernel
    CUdeviceptr d_a_ptx, d_b_ptx, d_c_ptx;
    CHECK(cuMemAlloc(&d_a_ptx, size_a * sizeof(Complex)));
    CHECK(cuMemAlloc(&d_b_ptx, size_b * sizeof(Complex)));
    CHECK(cuMemAlloc(&d_c_ptx, size_c * sizeof(Complex)));
    
    // Copy data to device
    CHECK(cuMemcpyHtoD(d_a_ptx, h_a, size_a * sizeof(Complex)));
    CHECK(cuMemcpyHtoD(d_b_ptx, h_b, size_b * sizeof(Complex)));
    
    // Setup kernel parameters
    int block_size = 256;
    int grid_size = (size_c + block_size - 1) / block_size;
    
    void* args[] = {
        &d_c_ptx,
        &d_a_ptx,
        &d_b_ptx,
        &M,
        &N,
        &K
    };
    
    // Launch PTX kernel
    printf("Launching PTX kernel: grid(%d), block(%d)\n", grid_size, block_size);
    
    // Create CUDA events for timing
    cudaEvent_t start_ptx, stop_ptx;
    CHECK_CUDA(cudaEventCreate(&start_ptx));
    CHECK_CUDA(cudaEventCreate(&stop_ptx));
    
    CHECK_CUDA(cudaEventRecord(start_ptx));
    
    CHECK(cuLaunchKernel(
        kernel,
        grid_size, 1, 1,      // grid dims
        block_size, 1, 1,     // block dims
        0,                    // shared memory
        0,                    // stream
        args,
        nullptr
    ));
    
    CHECK(cuCtxSynchronize());
    CHECK_CUDA(cudaEventRecord(stop_ptx));
    CHECK_CUDA(cudaEventSynchronize(stop_ptx));
    
    float time_ptx = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_ptx, start_ptx, stop_ptx));
    
    // Copy result back
    CHECK(cuMemcpyDtoH(h_c_ptx, d_c_ptx, size_c * sizeof(Complex)));
    
    printf("PTX kernel completed in %.3f ms\n", time_ptx);
    
    // ========================================================================
    // Method 2: Standard CUDA Runtime Kernel
    // ========================================================================
    
    printf("\n--- Running CUDA Runtime Kernel ---\n");
    
    // Allocate device memory for CUDA kernel
    Complex *d_a_cuda, *d_b_cuda, *d_c_cuda;
    CHECK_CUDA(cudaMalloc(&d_a_cuda, size_a * sizeof(Complex)));
    CHECK_CUDA(cudaMalloc(&d_b_cuda, size_b * sizeof(Complex)));
    CHECK_CUDA(cudaMalloc(&d_c_cuda, size_c * sizeof(Complex)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_a_cuda, h_a, size_a * sizeof(Complex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_cuda, h_b, size_b * sizeof(Complex), cudaMemcpyHostToDevice));
    
    // Launch CUDA kernel
    printf("Launching CUDA kernel: grid(%d), block(%d)\n", grid_size, block_size);
    
    cudaEvent_t start_cuda, stop_cuda;
    CHECK_CUDA(cudaEventCreate(&start_cuda));
    CHECK_CUDA(cudaEventCreate(&stop_cuda));
    
    CHECK_CUDA(cudaEventRecord(start_cuda));
    
    cuda_mul_matrix_complex<<<grid_size, block_size>>>(d_c_cuda, d_a_cuda, d_b_cuda, M, N, K);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop_cuda));
    CHECK_CUDA(cudaEventSynchronize(stop_cuda));
    
    float time_cuda = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_cuda, start_cuda, stop_cuda));
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_c_cuda, d_c_cuda, size_c * sizeof(Complex), cudaMemcpyDeviceToHost));
    
    printf("CUDA kernel completed in %.3f ms\n", time_cuda);
    
    // ========================================================================
    // Compare Results
    // ========================================================================
    
    printf("\n--- Comparing Results ---\n");
    
    print_complex_matrix("Result (PTX)", h_c_ptx, M, N);
    print_complex_matrix("Result (CUDA)", h_c_cuda, M, N);
    
    bool match = compare_results(h_c_ptx, h_c_cuda, size_c);
    
    if (match) {
        printf("\n✓ SUCCESS: PTX and CUDA results match!\n");
    } else {
        printf("\n✗ FAILURE: Results do not match!\n");
    }
    
    printf("\nPerformance Comparison:\n");
    printf("  PTX Kernel:  %.3f ms\n", time_ptx);
    printf("  CUDA Kernel: %.3f ms\n", time_cuda);
    printf("  Speedup:     %.2fx %s\n", 
           time_ptx / time_cuda,
           time_ptx < time_cuda ? "(PTX faster)" : "(CUDA faster)");
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    // PTX cleanup
    cuMemFree(d_a_ptx);
    cuMemFree(d_b_ptx);
    cuMemFree(d_c_ptx);
    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(dev);
    
    // CUDA cleanup
    cudaFree(d_a_cuda);
    cudaFree(d_b_cuda);
    cudaFree(d_c_cuda);
    cudaEventDestroy(start_ptx);
    cudaEventDestroy(stop_ptx);
    cudaEventDestroy(start_cuda);
    cudaEventDestroy(stop_cuda);
    
    // Host cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_ptx;
    delete[] h_c_cuda;
    
    return 0;
}
