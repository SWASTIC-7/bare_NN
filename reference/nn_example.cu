// Minimal 2-hidden-layer neural network running on GPU using cuBLAS + custom kernels.
// Builds with: nvcc -O2 -lcublas -lcurand -o nn_example src/nn_example.cu
// Generates PTX: nvcc -ptx src/nn_example.cu -o ptx/nn_example.ptx

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

// Simple error-checking macros
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)
#define CUBLAS_CHECK(call) do { cublasStatus_t s=(call); if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"CUBLAS error %d\n",s); exit(1);} } while(0)
#define CURAND_CHECK(call) do { curandStatus_t r=(call); if(r!=CURAND_STATUS_SUCCESS){fprintf(stderr,"CURAND error %d\n",r); exit(1);} } while(0)

// Elementwise ReLU and derivative
__global__ void relu_forward(float* A, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { A[i] = fmaxf(A[i], 0.0f); }
}
__global__ void relu_backward(float* grad, const float* A, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { grad[i] = (A[i] > 0.0f) ? grad[i] : 0.0f; }
}

// Softmax per-row and compute cross-entropy loss; N = batch, C = classes, stride = C
__global__ void softmax_and_loss(float* logits, const int* labels, float* probs, float* loss_out, int N, int C) {
    int r = blockIdx.x; // one block per sample (assumes gridDim.x == N)
    int tid = threadIdx.x;
    extern __shared__ float sdata[]; // size C
    if (r >= N) return;
    // load logits
    float maxv = -1e30f;
    for (int j = tid; j < C; j += blockDim.x) {
        float v = logits[r*C + j];
        if (v > maxv) maxv = v;
        sdata[j] = v; // temporarily store
    }
    __syncthreads();
    // find max across threads
    // simple second pass to find max (C small - e.g., 10)
    if (tid == 0) {
        float m = -1e30f;
        for (int j = 0; j < C; ++j) if (sdata[j] > m) m = sdata[j];
        sdata[0] = m; // reuse index 0 for max
    }
    __syncthreads();
    float max_all = sdata[0];
    // exponentiate & sum
    float sum = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        float ex = expf(logits[r*C + j] - max_all);
        probs[r*C + j] = ex;
        sum += ex;
    }
    // reduce sum across threads
    // store partials in sdata (reuse indices)
    float partial = sum;
    __shared__ float ssum[32]; // support blockDim.x <= 32
    if (tid < 32) ssum[tid] = 0.0f;
    __syncthreads();
    atomicAdd(&ssum[0], partial);
    __syncthreads();
    float total = ssum[0];
    // normalize and compute loss contribution
    for (int j = tid; j < C; j += blockDim.x) {
        float p = probs[r*C + j] / total;
        probs[r*C + j] = p;
    }
    __syncthreads();
    if (tid == 0) {
        int lb = labels[r];
        float p = probs[r*C + lb];
        float l = -logf(fmaxf(p, 1e-8f));
        loss_out[r] = l;
    }
}

// subtract labels from probs to produce upstream gradient (probs - onehot)
__global__ void softmax_grad(float* probs, const int* labels, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx < total) {
        int r = idx / C;
        int c = idx % C;
        float v = probs[idx];
        if (c == labels[r]) v -= 1.0f;
        probs[idx] = v;
    }
}

// Utility: fill device array with zeros
void gpu_zero(float* d, size_t n) { CUDA_CHECK(cudaMemset(d, 0, n*sizeof(float))); }

// SGD update: W = W - lr * dW
__global__ void sgd_update(float* W, const float* dW, float lr, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { W[i] -= lr * dW[i]; }
}

int main() {
    // Network sizes (small default for demonstration)
    const int input_dim = 784; // e.g., flattened 28x28 images
    const int hidden1 = 128;
    const int hidden2 = 64;
    const int output_dim = 10;
    const int batch = 64;
    const int epochs = 5;
    const float lr = 0.01f;

    // Create cuBLAS and cuRAND
    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));
    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, 1234ULL));

    // Allocate weight matrices on device: W1 (input x h1), W2 (h1 x h2), W3 (h2 x out)
    float *W1, *W2, *W3;
    CUDA_CHECK(cudaMalloc(&W1, input_dim*hidden1*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W2, hidden1*hidden2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W3, hidden2*output_dim*sizeof(float)));
    // Allocate activations and gradients
    float *X, *Z1, *Z2, *Logits; // X: batch x input, Z1: batch x h1, Z2: batch x h2, Logits: batch x out
    CUDA_CHECK(cudaMalloc(&X, batch*input_dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Z1, batch*hidden1*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Z2, batch*hidden2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Logits, batch*output_dim*sizeof(float)));
    // probs inplace on Logits during forward
    
    // Allocate gradient buffers
    float *dW1, *dW2, *dW3; // gradients for weights
    float *dZ1, *dZ2, *dLogits; // gradients for activations
    float *Z1_pre, *Z2_pre; // pre-activation values for ReLU backward
    CUDA_CHECK(cudaMalloc(&dW1, input_dim*hidden1*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW2, hidden1*hidden2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW3, hidden2*output_dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dZ1, batch*hidden1*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dZ2, batch*hidden2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dLogits, batch*output_dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Z1_pre, batch*hidden1*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Z2_pre, batch*hidden2*sizeof(float)));

    // label storage
    int* d_labels;
    CUDA_CHECK(cudaMalloc(&d_labels, batch*sizeof(int)));

    // Random initialization
    CURAND_CHECK(curandGenerateNormal(rng, W1, input_dim*hidden1, 0.0f, 0.1f));
    CURAND_CHECK(curandGenerateNormal(rng, W2, hidden1*hidden2, 0.0f, 0.1f));
    CURAND_CHECK(curandGenerateNormal(rng, W3, hidden2*output_dim, 0.0f, 0.1f));

    // Synthetic dataset in-device: random inputs and random labels
    CURAND_CHECK(curandGenerateNormal(rng, X, batch*input_dim, 0.0f, 1.0f));
    // random labels between 0 and output_dim-1
    std::vector<int> h_labels(batch);
    for (int i=0;i<batch;i++) h_labels[i] = rand() % output_dim;
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), batch*sizeof(int), cudaMemcpyHostToDevice));

    // Temporary arrays for loss
    float* d_loss; CUDA_CHECK(cudaMalloc(&d_loss, batch*sizeof(float)));

    // Training loop (very small, for demonstration)
    // Note: matrices are treated as row-major; cublas uses column-major, so we swap A/B and transpose flags accordingly.
    for (int ep=0; ep<epochs; ++ep) {
        // Forward: Z1 = X * W1
        const float alpha = 1.0f, beta = 0.0f;
        // compute Z1 = X * W1 (row-major: batch x input_dim) * (input_dim x hidden1) = (batch x hidden1)
        // cuBLAS col-major view: compute Z1^T = W1^T * X^T
        CUBLAS_CHECK(cublasSgemm(blas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden1, batch, input_dim,
            &alpha,
            W1, hidden1,   // W1 row-major (input_dim x hidden1), lda = num_cols = hidden1
            X, input_dim,  // X row-major (batch x input_dim), ldb = num_cols = input_dim
            &beta,
            Z1, hidden1)); // Z1 row-major (batch x hidden1), ldc = num_cols = hidden1
        // Save pre-activation and apply ReLU
        int N1 = batch*hidden1;
        CUDA_CHECK(cudaMemcpy(Z1_pre, Z1, N1*sizeof(float), cudaMemcpyDeviceToDevice));
        relu_forward<<<(N1+255)/256,256>>>(Z1,N1);

        // Z2 = Z1 * W2 (row-major: batch x hidden1) * (hidden1 x hidden2) = (batch x hidden2)
        CUBLAS_CHECK(cublasSgemm(blas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden2, batch, hidden1,
            &alpha,
            W2, hidden2,   // W2 row-major (hidden1 x hidden2), lda = hidden2
            Z1, hidden1,   // Z1 row-major (batch x hidden1), ldb = hidden1
            &beta,
            Z2, hidden2)); // Z2 row-major (batch x hidden2), ldc = hidden2
        int N2 = batch*hidden2;
        CUDA_CHECK(cudaMemcpy(Z2_pre, Z2, N2*sizeof(float), cudaMemcpyDeviceToDevice));
        relu_forward<<<(N2+255)/256,256>>>(Z2,N2);

        // Logits = Z2 * W3 (row-major: batch x hidden2) * (hidden2 x output_dim) = (batch x output_dim)
        CUBLAS_CHECK(cublasSgemm(blas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            output_dim, batch, hidden2,
            &alpha,
            W3, output_dim,   // W3 row-major (hidden2 x output_dim), lda = output_dim
            Z2, hidden2,      // Z2 row-major (batch x hidden2), ldb = hidden2
            &beta,
            Logits, output_dim)); // Logits row-major (batch x output_dim), ldc = output_dim

        // Softmax + loss
        int threads = (output_dim < 32) ? output_dim : 32;
        softmax_and_loss<<<batch, threads, output_dim * sizeof(float)>>>(Logits, d_labels, Logits, d_loss, batch, output_dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute average loss
        std::vector<float> h_loss(batch);
        CUDA_CHECK(cudaMemcpy(h_loss.data(), d_loss, batch*sizeof(float), cudaMemcpyDeviceToHost));
        float loss = 0.0f; for (int i=0;i<batch;i++) loss += h_loss[i]; loss /= batch;
        printf("Epoch %d loss=%f\n", ep, loss);

        // ============ BACKWARD PASS ============
        // Compute dLogits = probs - onehot (stored in dLogits)
        CUDA_CHECK(cudaMemcpy(dLogits, Logits, batch*output_dim*sizeof(float), cudaMemcpyDeviceToDevice));
        int total = batch*output_dim;
        softmax_grad<<<(total+255)/256,256>>>(dLogits, d_labels, batch, output_dim);

        // Gradient for W3: dW3 = Z2^T * dLogits / batch
        // Z2: (batch x hidden2), dLogits: (batch x output_dim) -> dW3: (hidden2 x output_dim)
        float scale = 1.0f / batch;
        CUBLAS_CHECK(cublasSgemm(blas,
            CUBLAS_OP_N, CUBLAS_OP_T,
            output_dim, hidden2, batch,
            &scale,
            dLogits, output_dim, // dLogits row-major (batch x output_dim), lda = output_dim
            Z2, hidden2,         // Z2 row-major (batch x hidden2), transposed, ldb = hidden2
            &beta,
            dW3, output_dim));   // dW3 row-major (hidden2 x output_dim), ldc = output_dim

        // Gradient for Z2: dZ2 = dLogits * W3^T
        // dLogits: (batch x output_dim), W3: (hidden2 x output_dim) -> dZ2: (batch x hidden2)
        CUBLAS_CHECK(cublasSgemm(blas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            hidden2, batch, output_dim,
            &alpha,
            W3, output_dim,      // W3 row-major (hidden2 x output_dim), transposed, lda = output_dim
            dLogits, output_dim, // dLogits row-major (batch x output_dim), ldb = output_dim
            &beta,
            dZ2, hidden2));      // dZ2 row-major (batch x hidden2), ldc = hidden2
        // Apply ReLU gradient
        relu_backward<<<(N2+255)/256,256>>>(dZ2, Z2_pre, N2);

        // Gradient for W2: dW2 = Z1^T * dZ2 / batch
        // Z1: (batch x hidden1), dZ2: (batch x hidden2) -> dW2: (hidden1 x hidden2)
        CUBLAS_CHECK(cublasSgemm(blas,
            CUBLAS_OP_N, CUBLAS_OP_T,
            hidden2, hidden1, batch,
            &scale,
            dZ2, hidden2,    // dZ2 row-major (batch x hidden2), lda = hidden2
            Z1, hidden1,     // Z1 row-major (batch x hidden1), transposed, ldb = hidden1
            &beta,
            dW2, hidden2));  // dW2 row-major (hidden1 x hidden2), ldc = hidden2

        // Gradient for Z1: dZ1 = dZ2 * W2^T
        // dZ2: (batch x hidden2), W2: (hidden1 x hidden2) -> dZ1: (batch x hidden1)
        CUBLAS_CHECK(cublasSgemm(blas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            hidden1, batch, hidden2,
            &alpha,
            W2, hidden2,   // W2 row-major (hidden1 x hidden2), transposed, lda = hidden2
            dZ2, hidden2,  // dZ2 row-major (batch x hidden2), ldb = hidden2
            &beta,
            dZ1, hidden1)); // dZ1 row-major (batch x hidden1), ldc = hidden1
        // Apply ReLU gradient
        relu_backward<<<(N1+255)/256,256>>>(dZ1, Z1_pre, N1);

        // Gradient for W1: dW1 = X^T * dZ1 / batch
        // X: (batch x input_dim), dZ1: (batch x hidden1) -> dW1: (input_dim x hidden1)
        CUBLAS_CHECK(cublasSgemm(blas,
            CUBLAS_OP_N, CUBLAS_OP_T,
            hidden1, input_dim, batch,
            &scale,
            dZ1, hidden1,   // dZ1 row-major (batch x hidden1), lda = hidden1
            X, input_dim,   // X row-major (batch x input_dim), transposed, ldb = input_dim
            &beta,
            dW1, hidden1)); // dW1 row-major (input_dim x hidden1), ldc = hidden1

        // ============ SGD UPDATE ============
        // Update weights: W = W - lr * dW
        sgd_update<<<(input_dim*hidden1+255)/256,256>>>(W1, dW1, lr, input_dim*hidden1);
        sgd_update<<<(hidden1*hidden2+255)/256,256>>>(W2, dW2, lr, hidden1*hidden2);
        sgd_update<<<(hidden2*output_dim+255)/256,256>>>(W3, dW3, lr, hidden2*output_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Cleanup
    cudaFree(W1); cudaFree(W2); cudaFree(W3);
    cudaFree(dW1); cudaFree(dW2); cudaFree(dW3);
    cudaFree(X); cudaFree(Z1); cudaFree(Z2); cudaFree(Logits);
    cudaFree(dZ1); cudaFree(dZ2); cudaFree(dLogits);
    cudaFree(Z1_pre); cudaFree(Z2_pre);
    cudaFree(d_labels); cudaFree(d_loss);
    cublasDestroy(blas);
    curandDestroyGenerator(rng);
    return 0;
}
