import math
import time

import torch
import triton
import triton.language as tl


BT = 16
TK = 16
WM = 2
WN = 16


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & ((k0 + offs_k[None, :]) < K)
        b_mask = ((k0 + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    device = "cuda"
    n = 1024



    a = torch.sin(torch.arange(n * n, device=device, dtype=torch.float32)).reshape(n, n)
    b = torch.cos(torch.arange(n * n, device=device, dtype=torch.float32)).reshape(n, n)
    c = torch.empty((n, n), device=device, dtype=torch.float32)

    grid = (triton.cdiv(n, BT), triton.cdiv(n, BT))

    matmul_kernel[grid](
        a,
        b,
        c,
        n,
        n,
        n,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BT,
        BLOCK_N=BT,
        BLOCK_K=TK,
        num_warps=4,
        num_stages=3,
    )
    torch.cuda.synchronize()

    reps = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(reps):
        matmul_kernel[grid](
            a,
            b,
            c,
            n,
            n,
            n,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=BT,
            BLOCK_N=BT,
            BLOCK_K=TK,
            num_warps=4,
            num_stages=3,
        )
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / reps

    ref = torch.matmul(a[:64, :].double(), b[:, :64].double())
    out = c[:64, :64].double()
    diff = torch.abs(ref - out)
    max_error = diff.max().item()

    gflops = (2.0 * n * n * n) / (ms * 1e6)

   
    print(gflops)


if __name__ == "__main__":
    run()
