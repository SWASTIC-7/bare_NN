#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// Vector Elementwise Operations
// ============================================================================

// Vector addition: c[i] = a[i] + b[i]
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Vector subtraction: c[i] = a[i] - b[i]
__global__ void vectorSub(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

// Vector elementwise multiplication (Hadamard product): c[i] = a[i] * b[i]
__global__ void vectorMul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Vector elementwise division: c[i] = a[i] / b[i]
__global__ void vectorDiv(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

// Scalar addition: c[i] = a[i] + scalar
__global__ void vectorAddScalar(const float* a, float scalar, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + scalar;
    }
}

// Scalar multiplication: c[i] = a[i] * scalar
__global__ void vectorMulScalar(const float* a, float scalar, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * scalar;
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

// ReLU: c[i] = max(0, a[i])
__global__ void vectorReLU(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fmaxf(0.0f, a[idx]);
    }
}

// ReLU derivative: c[i] = a[i] > 0 ? 1 : 0
__global__ void vectorReLUGradient(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] > 0.0f ? 1.0f : 0.0f;
    }
}

// Sigmoid: c[i] = 1 / (1 + exp(-a[i]))
__global__ void vectorSigmoid(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = 1.0f / (1.0f + expf(-a[idx]));
    }
}

// Tanh: c[i] = tanh(a[i])
__global__ void vectorTanh(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = tanhf(a[idx]);
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

// Sum reduction using shared memory
__global__ void vectorSum(const float* a, float* result, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? a[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// Mean: sum(a[i]) / n
__global__ void vectorMean(const float* a, float* result, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? a[idx] : 0.0f;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0] / n);
    }
}

// Max reduction
__global__ void vectorMax(const float* a, float* result, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? a[idx] : -INFINITY;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // Atomic max operation
        float old = *result;
        float assumed;
        do {
            assumed = old;
            old = atomicCAS((int*)result, 
                           __float_as_int(assumed),
                           __float_as_int(fmaxf(assumed, sdata[0])));
        } while (assumed != old);
    }
}

// Min reduction
__global__ void vectorMin(const float* a, float* result, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? a[idx] : INFINITY;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float old = *result;
        float assumed;
        do {
            assumed = old;
            old = atomicCAS((int*)result, 
                           __float_as_int(assumed),
                           __float_as_int(fminf(assumed, sdata[0])));
        } while (assumed != old);
    }
}

// ============================================================================
// Dot Product and Norms
// ============================================================================

// Dot product: sum(a[i] * b[i])
__global__ void vectorDot(const float* a, const float* b, float* result, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// L2 norm squared: sum(a[i]^2)
__global__ void vectorNormL2Squared(const float* a, float* result, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < n) ? a[idx] : 0.0f;
    sdata[tid] = val * val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}


// ============================================================================
// Utility Functions
// ============================================================================

// Fill vector with constant value
__global__ void vectorFill(float* a, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = value;
    }
}

// Copy vector
__global__ void vectorCopy(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Clamp values: c[i] = clamp(a[i], min_val, max_val)
__global__ void vectorClamp(const float* a, float* c, float min_val, float max_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fminf(fmaxf(a[idx], min_val), max_val);
    }
}

// ============================================================================
// Test/Demo Function
// ============================================================================

int main() {
    const int N = 1024;
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];
    
    // Initialize test data
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c, *d_result;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Test vector addition
    printf("Testing Vector Operations:\n");
    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Add: a[0] + b[0] = %.2f + %.2f = %.2f\n", h_a[0], h_b[0], h_c[0]);
    
    // Test vector sum
    float h_sum = 0.0f;
    cudaMemcpy(d_result, &h_sum, sizeof(float), cudaMemcpyHostToDevice);
    vectorSum<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_a, d_result, N);
    cudaMemcpy(&h_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sum: %.2f\n", h_sum);
    
    // Test vector mean
    float h_mean = 0.0f;
    cudaMemcpy(d_result, &h_mean, sizeof(float), cudaMemcpyHostToDevice);
    vectorMean<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_a, d_result, N);
    cudaMemcpy(&h_mean, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Mean: %.2f\n", h_mean);
    
    // Test vector max
    float h_max = -INFINITY;
    cudaMemcpy(d_result, &h_max, sizeof(float), cudaMemcpyHostToDevice);
    vectorMax<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_a, d_result, N);
    cudaMemcpy(&h_max, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Max: %.2f\n", h_max);
    
    // Test ReLU
    vectorReLU<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_c, N);
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("ReLU: max(0, %.2f) = %.2f\n", h_a[0], h_c[0]);
    
    // Test dot product
    float h_dot = 0.0f;
    cudaMemcpy(d_result, &h_dot, sizeof(float), cudaMemcpyHostToDevice);
    vectorDot<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_a, d_b, d_result, N);
    cudaMemcpy(&h_dot, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Dot product: %.2f\n", h_dot);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_result);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    printf("\nAll tests completed!\n");
    return 0;
}
