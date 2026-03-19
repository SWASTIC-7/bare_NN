import torch
import triton
import triton.language as tl

BT = 16
TK = 32
WTX = 16
WTY = 2
TTX = 1
TTY = 1


@triton.jit
def thread_tiled_kernel(
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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")

    n = 1024
    device = "cuda"

    print(f"Config: BT={BT} TK={TK} WTX={WTX} WTY={WTY} TTX={TTX} TTY={TTY}")

    a = torch.sin(torch.arange(n * n, device=device, dtype=torch.float32)).reshape(n, n)
    b = torch.cos(torch.arange(n * n, device=device, dtype=torch.float32)).reshape(n, n)
    c = torch.empty((n, n), device=device, dtype=torch.float32)

    grid = (triton.cdiv(n, BT), triton.cdiv(n, BT))

    thread_tiled_kernel[grid](
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
        num_warps=8,
        num_stages=3,
    )
    torch.cuda.synchronize()

    reps = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(reps):
        thread_tiled_kernel[grid](
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
            num_warps=8,
            num_stages=3,
        )
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / reps

    checksum = c.double().sum().item()
    ref = torch.matmul(a.double(), b.double())
    diff = (ref - c.double()).abs()
    max_error = diff.max().item()
    error_count = (diff > 1e-2).sum().item()

    gflops = (2.0 * n * n * n) / (ms * 1e6)

    print(f"Checksum:     {checksum:.10e}")
    print(f"Max error:    {max_error:.10e}")
    print(f"Error count:  {int(error_count)}")
    print(f"Time:         {ms:.3f} ms")
    print(f"Performance:  {gflops:.2f} GFLOPS")


if __name__ == "__main__":
    main()
