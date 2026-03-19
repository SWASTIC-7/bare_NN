import torch
import triton
import triton.language as tl
import math
import time

@triton.jit
def tiled_triton(
    C_ptr, A_ptr, B_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BT_M: tl.constexpr,
    BT_N: tl.constexpr,
    BT_K: tl.constexpr,
):
    block_row = tl.program_id(1)
    block_col = tl.program_id(0)

    row_offs = block_row * BT_M + tl.arange(0, BT_M)   # [BT_M]
    col_offs = block_col * BT_N + tl.arange(0, BT_N)   # [BT_N]

    acc = tl.zeros((BT_M, BT_N), dtype=tl.float32)

    for i in range(0, K, BT_K):
        k_offs = i + tl.arange(0, BT_K)                # [BT_K]

        # Load A tile: [BT_M, BT_K]
        a_ptrs = A_ptr + row_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (row_offs[:, None] < M) & (k_offs[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: [BT_K, BT_N]
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + col_offs[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (col_offs[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a_tile, b_tile)

    # Write C
    c_ptrs = C_ptr + row_offs[:, None] * stride_cm + col_offs[None, :] * stride_cn
    c_mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def run(BT_M=32, BT_N=32, BT_K=32):
    M, N, K = 1024, 1024, 1024
    device = "cuda"

    # --- init matrices with sin/cos ---
    idx_a = torch.arange(M * K, dtype=torch.float32, device=device)
    idx_b = torch.arange(K * N, dtype=torch.float32, device=device)
    A = torch.sin(idx_a).reshape(M, K)
    B = torch.cos(idx_b).reshape(K, N)
    C = torch.zeros((M, N), dtype=torch.float32, device=device)

    grid = ((N + BT_N - 1) // BT_N,
            (M + BT_M - 1) // BT_M)

    # --- warmup ---
    tiled_triton[grid](
        C, A, B,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BT_M=BT_M, BT_N=BT_N, BT_K=BT_K,
    )
    torch.cuda.synchronize()

    # --- timed run ---
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    tiled_triton[grid](
        C, A, B,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BT_M=BT_M, BT_N=BT_N, BT_K=BT_K,
    )
    end.record()
    torch.cuda.synchronize()

    ms     = start.elapsed_time(end)
    gflops = (2.0 * M * N * K) / (ms * 1e6)

    print(gflops)





if __name__ == "__main__":
    run(BT_M=32, BT_N=32, BT_K=32)