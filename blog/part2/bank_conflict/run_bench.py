"""Benchmark for the 'Shared-memory bank conflicts' section.

All three kernels store the A tile transposed and PAD its leading dim (LDA = BM+4)
to remove the transpose-store bank conflict:
  * back_conflict.ptx   -- vectorized PTX, padded (TT 4x4)
  * bank_conflict.cu    -- vectorized CUDA, padded (NVRTC per config)
  * thread_tiled_triton -- Triton (handles bank conflicts on its own)
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

# ---- PTX (padded, TT 4x4) ----
PAD = 4
BT_WT = [(btm, btn, btk, wtx, wty)
         for btm in (64, 128) for btn in (64, 128) for btk in (8, 16)
         for (wtx, wty) in ((8, 4), (4, 8), (16, 2), (32, 1))]
ptx_fn = cp.RawModule(path=str(HERE / "back_conflict.ptx")).get_function("thread_tiled_ptx")


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
    smem = (BT_K * (BT_M + PAD) + BT_K * BT_N) * 4
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

# ---- CUDA (padded, NVRTC per config) ----
cu_src = (HERE / "bank_conflict.cu").read_text()


def cuda_bench(cfg):
    BM, BN, BK, TM, TN = cfg
    if BK % 4 or BN % 4 or TM % 4 or TN % 4 or BM % TM or BN % TN:
        return None
    threads = (BM * BN) // (TM * TN)
    if threads % 32 or threads > 1024:
        return None
    gA, gB = BK // 4, BN // 4
    if threads % gA or threads % gB:
        return None
    strideA, strideB = threads // gA, threads // gB
    if strideA == 0 or strideB == 0 or BM % strideA or BK % strideB:
        return None
    if (BK * (BM + PAD) + BK * BN) * 4 > 48 * 1024:
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

# ---- chart ----
(HERE / "results.json").write_text(json.dumps(
    {"ptx": ptx_gflops, "cuda": cuda_gflops, "triton": tri_gflops,
     "cublas_fp32": cublas_gflops}, indent=2))
plot_bars({"CUDA": cuda_gflops, "PTX": ptx_gflops, "Triton": tri_gflops,
           "cuBLAS": cublas_gflops},
          out_path=str(HERE / "bank_conflict_all.png"))
print("wrote bank_conflict_all.png, results.json")
