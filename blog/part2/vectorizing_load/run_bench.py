"""Benchmark for the 'Vectorizing load' section.

Everything is vectorized (128-bit / float4) and autotuned:
  * vectorized.ptx           -- PTX with .v4.f32 loads/stores, A transposed (TT 4x4)
  * vectorized.cu            -- CUDA mirror (float4 loads, A transposed)
  * thread_tiled_triton.py   -- Triton (already vectorizes internally)
Also shows the scalar->vectorized gain for the PTX kernel.
Specs (same as part 1): 1024 x 1024 x 1024, NVIDIA GeForce RTX 3050.

Run:   python run_bench.py
Deps:  cupy, torch, triton, matplotlib, numpy
"""
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # reusable scripts live in part2/

import numpy as np
import cupy as cp

import bench_utils as bu
from autotune import autotune
from plot_benchmark import plot_bars
from reference_fp32 import TRITON_FP32, CUBLAS_FP32   # fixed FP32 refs (measured once)

M = N = K = 1024
A, B, C, ref = bu.make_problem(M, N, K)

BT_WT = [(btm, btn, btk, wtx, wty)
         for btm in (64, 128) for btn in (64, 128) for btk in (8, 16)
         for (wtx, wty) in ((8, 4), (4, 8), (16, 2), (32, 1))]


def make_ptx_bench(fn):
    """PTX kernels here are fixed at TT 4x4; sweep BT/WT only."""
    def bench(cfg):
        BT_M, BT_N, BT_K, WT_X, WT_Y = cfg
        TT_X = TT_Y = 4
        if BT_N % 4 or BT_K % 4:                 # v4 needs multiples of 4
            return None
        tpr, tpc = BT_N // TT_X, BT_M // TT_Y
        if tpr % WT_X or tpc % WT_Y:
            return None
        threads = tpr * tpc
        if threads % 32 or threads > 1024:
            return None
        smem = (BT_M * BT_K + BT_K * BT_N) * 4
        if smem > 48 * 1024:
            return None
        grid = ((N + BT_N - 1) // BT_N, (M + BT_M - 1) // BT_M, 1)
        block = (threads, 1, 1)
        args = (C, A, B, np.int32(M), np.int32(N), np.int32(K),
                np.int32(BT_M), np.int32(BT_N), np.int32(BT_K),
                np.int32(WT_X), np.int32(WT_Y), np.int32(TT_X), np.int32(TT_Y))
        launch = lambda: fn(grid, block, args, shared_mem=smem)
        C.fill(0)
        launch()
        cp.cuda.runtime.deviceSynchronize()
        if bu.rel_error(C, ref) > 1e-3:
            return None
        return bu.time_cupy(launch)
    return bench


# scalar 4x4 PTX (from the previous section) -- the "before"
scalar_fn = cp.RawModule(path=str(HERE.parent / "fixing_thread_tiling" / "new.ptx")).get_function("thread_tiled_ptx")
scfg, sms, _ = autotune(BT_WT, make_ptx_bench(scalar_fn), label="PTX-scalar")
ptx_scalar = bu.gflops(M, N, K, sms)
print(f"PTX scalar     best {scfg}: {sms:.3f} ms  {ptx_scalar:.1f} GFLOP/s")

# vectorized PTX -- the "after"
vec_fn = cp.RawModule(path=str(HERE / "vectorized.ptx")).get_function("thread_tiled_ptx")
vcfg, vms, _ = autotune(BT_WT, make_ptx_bench(vec_fn), label="PTX-vec")
ptx_vec = bu.gflops(M, N, K, vms)
print(f"PTX vectorized best {vcfg}: {vms:.3f} ms  {ptx_vec:.1f} GFLOP/s")

# vectorized CUDA
cu_src = (HERE / "vectorized.cu").read_text()


def cuda_bench(cfg):
    BM, BN, BK, TM, TN = cfg
    if BK % 4 or BN % 4 or TM % 4 or TN % 4:
        return None
    if BM % TM or BN % TN:
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
    opts = (f"-DBM={BM}", f"-DBN={BN}", f"-DBK={BK}", f"-DTM={TM}", f"-DTN={TN}")
    fn = cp.RawModule(code=cu_src, backend="nvrtc", options=opts).get_function("thread_tiled_matmul")
    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
    block = (threads, 1, 1)
    args = (np.int32(M), np.int32(N), np.int32(K), A, B, C)
    launch = lambda: fn(grid, block, args)
    C.fill(0)
    launch()
    cp.cuda.runtime.deviceSynchronize()
    if bu.rel_error(C, ref) > 1e-3:
        return None
    return bu.time_cupy(launch)


cuda_space = [(bm, bn, bk, tm, tn)
              for bm in (64, 128) for bn in (64, 128) for bk in (8, 16)
              for tm in (4, 8) for tn in (4, 8)]
ccfg, cms, _ = autotune(cuda_space, cuda_bench, label="CUDA-vec")
cuda_gflops = bu.gflops(M, N, K, cms)
print(f"CUDA vectorized best {ccfg}: {cms:.3f} ms  {cuda_gflops:.1f} GFLOP/s")

# Triton FP32 + cuBLAS FP32 : fixed references (measured once in reference_fp32.py)
tri_gflops, cublas_gflops = TRITON_FP32, CUBLAS_FP32

# ---------------------------------------------------------------------------
# charts + results
# ---------------------------------------------------------------------------
(HERE / "results.json").write_text(json.dumps({
    "ptx_scalar": ptx_scalar, "ptx_vectorized": ptx_vec,
    "cuda_vectorized": cuda_gflops, "triton": tri_gflops,
    "cublas_fp32": cublas_gflops,
}, indent=2))

plot_bars({"CUDA": cuda_gflops, "PTX": ptx_vec, "Triton": tri_gflops,
           "cuBLAS": cublas_gflops},
          out_path=str(HERE / "vectorized_all.png"))
plot_bars({"PTX scalar": ptx_scalar, "PTX vectorized": ptx_vec},
          out_path=str(HERE / "vectorized_gain.png"))

print("wrote vectorized_all.png, vectorized_gain.png, results.json")
