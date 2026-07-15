"""Fixed FP32 reference numbers for the Part-2 section charts.

Triton and cuBLAS are NOT optimized across the sections -- their code is identical
everywhere -- so we measure each ONCE here and reuse the same value as a constant
reference bar in every chart. Only PTX and CUDA are re-measured per section (those
are what we hand-optimize). This keeps the two reference bars from wobbling run to
run and makes every section chart directly comparable.

    from reference_fp32 import TRITON_FP32, CUBLAS_FP32

Re-measure whenever the machine/driver changes:   python reference_fp32.py
then paste the two printed numbers into the constants below.

Specs: 1024 x 1024 x 1024, NVIDIA GeForce RTX 3050, FP32 CUDA cores (no TF32).
"""

# --- measured once by running this file directly (see __main__) ---
TRITON_FP32 = 4871.0   # GFLOP/s  Triton tl.dot(input_precision="ieee"), autotuned
CUBLAS_FP32 = 4937.7   # GFLOP/s  torch.matmul FP32 (allow_tf32=False)


if __name__ == "__main__":
    import torch
    import triton
    import triton.language as tl

    M = N = K = 1024

    def gflops(ms):
        return 2 * M * N * K / (ms * 1e-3) / 1e9

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _matmul_kernel(
            A, B, C, M, N, K,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        width = GROUP_M * grid_n
        group_id = pid // width
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(grid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % width) // group_size_m
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc += tl.dot(a, b, input_precision="ieee")  # FP32 CUDA cores (no TF32)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask)

    def triton_matmul(A, B):
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        _matmul_kernel[grid](A, B, C, M, N, K,
                             A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                             C.stride(0), C.stride(1))
        return C

    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)

    Ct = triton_matmul(A, B)
    tri_err = float(torch.linalg.norm(Ct - A @ B) / torch.linalg.norm(A @ B))
    tri_ms = triton.testing.do_bench(lambda: triton_matmul(A, B))
    print(f"TRITON_FP32 = {gflops(tri_ms):.1f}   # {tri_ms:.3f} ms  (rel err {tri_err:.1e})")

    torch.backends.cuda.matmul.allow_tf32 = False
    cub_ms = triton.testing.do_bench(lambda: A @ B)
    print(f"CUBLAS_FP32 = {gflops(cub_ms):.1f}   # {cub_ms:.3f} ms")
