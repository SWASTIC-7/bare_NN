"""Benchmark for the 'Double buffering' section.

All three kernels double-buffer the shared tiles (register-staged prefetch overlaps
global-load latency with the FMAs):
  * buffering.ptx        -- double-buffered PTX (TT 4x4)
  * buffering.cu         -- double-buffered CUDA (NVRTC per config)
  * thread_tiled_triton  -- Triton (num_stages pipeline = double/multi buffering)
Specs (same as part 1): 1024 x 1024 x 1024, NVIDIA GeForce RTX 3050.

Run:   python run_bench.py     Deps: cupy, torch, triton, matplotlib, numpy
"""
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import cupy as cp
import bench_utils as bu
from autotune import autotune
from plot_benchmark import plot_bars
from reference_fp32 import TRITON_FP32, CUBLAS_FP32   # fixed FP32 refs (measured once)

M = N = K = 1024
A, B, C, ref = bu.make_problem(M, N, K)
PAD = 4

BT_WT = [(btm, btn, btk, wtx, wty)
         for btm in (64, 128) for btn in (64, 128) for btk in (8, 16)
         for (wtx, wty) in ((8, 4), (4, 8), (16, 2), (32, 1))]

ptx_fn = cp.RawModule(path=str(HERE / "buffering.ptx")).get_function("thread_tiled_ptx")


def ptx_bench(cfg):
    BT_M, BT_N, BT_K, WT_X, WT_Y = cfg
    TT_X = TT_Y = 4
    if BT_N % 4 or BT_K % 4:
        return None
    tpr, tpc = BT_N // TT_X, BT_M // TT_Y
    if tpr % WT_X or tpc % WT_Y:
        return None
    threads = tpr * tpc
    if threads % 32 or threads > 1024:
        return None
    a_vecs, b_vecs = BT_M * BT_K // 4, BT_K * BT_N // 4
    if threads < a_vecs or threads < b_vecs:              # <=1 float4 per thread
        return None
    smem = 2 * (BT_K * (BT_M + PAD) + BT_K * BT_N) * 4     # two buffers
    if smem > 48 * 1024:
        return None
    grid = ((N + BT_N - 1) // BT_N, (M + BT_M - 1) // BT_M, 1)
    block = (threads, 1, 1)
    args = (C, A, B, np.int32(M), np.int32(N), np.int32(K),
            np.int32(BT_M), np.int32(BT_N), np.int32(BT_K),
            np.int32(WT_X), np.int32(WT_Y), np.int32(TT_X), np.int32(TT_Y))
    launch = lambda: ptx_fn(grid, block, args, shared_mem=smem)
    C.fill(0); launch(); cp.cuda.runtime.deviceSynchronize()
    if bu.rel_error(C, ref) > 1e-3:
        return None
    return bu.time_cupy(launch)


pcfg, pms, _ = autotune(BT_WT, ptx_bench, label="PTX")
ptx_gflops = bu.gflops(M, N, K, pms)
print(f"PTX  best {pcfg}: {pms:.3f} ms  {ptx_gflops:.1f} GFLOP/s")

# ---- CUDA (double-buffered, NVRTC per config) ----
cu_src = (HERE / "buffering.cu").read_text()


def cuda_bench(cfg):
    BM, BN, BK, TM, TN = cfg
    if BK % 4 or BN % 4 or TM % 4 or TN % 4 or BM % TM or BN % TN:
        return None
    threads = (BM * BN) // (TM * TN)
    if threads % 32 or threads > 1024:
        return None
    a_vecs, b_vecs = BM * BK // 4, BK * BN // 4
    if threads < a_vecs or threads < b_vecs:              # <=1 float4 per thread
        return None
    if 2 * (BK * (BM + PAD) + BK * BN) * 4 > 48 * 1024:
        return None
    opts = (f"-DBM={BM}", f"-DBN={BN}", f"-DBK={BK}", f"-DTM={TM}", f"-DTN={TN}")
    fn = cp.RawModule(code=cu_src, backend="nvrtc", options=opts).get_function("thread_tiled_matmul")
    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
    block = (threads, 1, 1)
    args = (np.int32(M), np.int32(N), np.int32(K), A, B, C)
    launch = lambda: fn(grid, block, args)
    C.fill(0); launch(); cp.cuda.runtime.deviceSynchronize()
    if bu.rel_error(C, ref) > 1e-3:
        return None
    return bu.time_cupy(launch)


cuda_space = [(bm, bn, bk, tm, tn)
              for bm in (64, 128) for bn in (64, 128) for bk in (8, 16)
              for tm in (4, 8) for tn in (4, 8)]
ccfg, cms, _ = autotune(cuda_space, cuda_bench, label="CUDA")
cuda_gflops = bu.gflops(M, N, K, cms)
print(f"CUDA best {ccfg}: {cms:.3f} ms  {cuda_gflops:.1f} GFLOP/s")

# ---- Triton FP32 + cuBLAS FP32 : fixed references (measured once in reference_fp32.py) ----
tri_gflops, cublas_gflops = TRITON_FP32, CUBLAS_FP32


# ---- compiled-tile double buffering : 8x4 thread tile baked into the PTX (step 5) ----
def bench_compiled_8x4():
    # baked config: BT_M=64 BT_N=128 BT_K=8 WT_X=8 WT_Y=4 TT_X=4 TT_Y=8 (8x4 tile, 32 accumulators)
    BT_M, BT_N, BT_K, WT_X, WT_Y, TT_X, TT_Y = 64, 128, 8, 8, 4, 4, 8
    smem = 2 * (BT_K * (BT_M + PAD) + BT_K * BT_N) * 4     # two buffers, lda = BT_M+PAD
    threads = (BT_M // TT_Y) * (BT_N // TT_X)              # = 256
    grid = ((N + BT_N - 1) // BT_N, (M + BT_M - 1) // BT_M, 1)
    block = (threads, 1, 1)
    fn = cp.RawModule(path=str(HERE / "compiled_buffering_8x4.ptx")).get_function("thread_tiled_ptx")
    args = (C, A, B, np.int32(M), np.int32(N), np.int32(K),
            np.int32(BT_M), np.int32(BT_N), np.int32(BT_K),
            np.int32(WT_X), np.int32(WT_Y), np.int32(TT_X), np.int32(TT_Y))
    launch = lambda: fn(grid, block, args, shared_mem=smem)
    C.fill(0); launch(); cp.cuda.runtime.deviceSynchronize()
    assert bu.rel_error(C, ref) <= 1e-3, "compiled 8x4 produced wrong result"
    return bu.gflops(M, N, K, bu.time_cupy(launch))


compiled_gflops = bench_compiled_8x4()
print(f"PTX compiled 8x4 : {compiled_gflops:.1f} GFLOP/s")

# ---- charts ----
(HERE / "results.json").write_text(json.dumps(
    {"ptx": ptx_gflops, "ptx_compiled_8x4": compiled_gflops, "cuda": cuda_gflops,
     "triton": tri_gflops, "cublas_fp32": cublas_gflops}, indent=2))
# runtime-tile double buffering (step 4)
plot_bars({"CUDA": cuda_gflops, "PTX": ptx_gflops, "Triton": tri_gflops,
           "cuBLAS": cublas_gflops},
          out_path=str(HERE / "double_buffering_all.png"))
# compiled-tile double buffering (step 5): PTX bar is the baked 8x4 kernel
plot_bars({"CUDA": cuda_gflops, "PTX": compiled_gflops, "Triton": tri_gflops,
           "cuBLAS": cublas_gflops},
          out_path=str(HERE / "compiled_buffering_all.png"))
print("wrote double_buffering_all.png, compiled_buffering_all.png, results.json")
