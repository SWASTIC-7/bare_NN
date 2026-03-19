import torch
import triton
import triton.language as tl
import time

M = N = K = 1024

A = torch.sin(torch.arange(M*K, device="cuda")).reshape(M, K).float()
B = torch.cos(torch.arange(K*N, device="cuda")).reshape(K, N).float()
C = torch.zeros((M, N), device="cuda")

@triton.jit
def kernel(A, B, C, M, N, K):
    pid = tl.program_id(0)
    row = pid // N
    col = pid % N

    acc = 0.0

    for k in range(K):
        a = tl.load(A + row*K + k)
        b = tl.load(B + k*N + col)
        acc += a * b

    tl.store(C + row*N + col, acc)

grid = (M*N,)

# warmup (compilation happens here)
kernel[grid](A, B, C, M, N, K)
torch.cuda.synchronize()

start = time.time()

kernel[grid](A, B, C, M, N, K)

torch.cuda.synchronize()
end = time.time()

ms = (end - start) * 1000
gflops = (2*M*N*K) / (ms * 1e6)

print( gflops)