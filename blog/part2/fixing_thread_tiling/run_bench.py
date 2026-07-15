"""Benchmark for the 'Fixing thread tiling' section.

Compares thread-tiled matmul, each autotuned:
  * old.ptx                       -- runtime-TT PTX, accumulators in LOCAL memory
  * new_2x2 / new / new_8x8 .ptx  -- PTX with TT fixed in REGISTERS (2x2, 4x4, 8x8)
  * thread_tiled.cu               -- CUDA, tile params swept (NVRTC per config)
  * thread_tiled_triton           -- Triton, its own @triton.autotune

Shows which thread-tile size the register path prefers, then the overall picture.
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

BT_WT = [
    (btm, btn, btk, wtx, wty)
    for btm in (64, 128) for btn in (64, 128) for btk in (8, 16)
    for (wtx, wty) in ((8, 4), (4, 8), (16, 2), (32, 1))
]


def make_ptx_bench(fn):
    """Same launch logic for every PTX kernel (old + register variants)."""
    def bench(cfg):
        BT_M, BT_N, BT_K, WT_X, WT_Y, TT_X, TT_Y = cfg
        if WT_X * WT_Y != 32:
            return None
        if BT_M % TT_Y or BT_N % TT_X:
            return None
        if TT_X > 8 or TT_Y > 8:                    # local-memory cap in old.ptx
            return None
        tpr, tpc = BT_N // TT_X, BT_M // TT_Y
        if tpr % WT_X or tpc % WT_Y:
            return None
        threads = tpr * tpc
        if threads == 0 or threads % 32 or threads > 1024:
            return None
        smem = (BT_M * BT_K + BT_K * BT_N) * 4
        if smem > 48 * 1024:
            return None

        grid = ((N + BT_N - 1) // BT_N, (M + BT_M - 1) // BT_M, 1)
        block = (threads, 1, 1)
        args = (C, A, B,
                np.int32(M), np.int32(N), np.int32(K),
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


# 1) old.ptx: local-memory accumulators (sweep TT too)
old_fn = cp.RawModule(path=str(HERE / "old.ptx")).get_function("thread_tiled_ptx")
old_space = [(*bw, tx, ty) for bw in BT_WT for tx in (2, 4, 8) for ty in (2, 4, 8)]
old_cfg, old_ms, _ = autotune(old_space, make_ptx_bench(old_fn), label="PTX-local")
ptx_local = bu.gflops(M, N, K, old_ms)
print(f"PTX local  best {old_cfg}: {old_ms:.3f} ms  {ptx_local:.1f} GFLOP/s")

# 2) register variants: which thread-tile size wins?
reg_kernels = [("2x2", "new_2x2.ptx", 2, 2),
               ("4x4", "new.ptx", 4, 4),
               ("8x8", "new_8x8.ptx", 8, 8)]
reg_gflops = {}
best_reg = None
for name, fname, tx, ty in reg_kernels:
    fn = cp.RawModule(path=str(HERE / fname)).get_function("thread_tiled_ptx")
    space = [(*bw, tx, ty) for bw in BT_WT]
    cfg, ms, _ = autotune(space, make_ptx_bench(fn), label=f"PTX-{name}")
    g = bu.gflops(M, N, K, ms)
    reg_gflops[name] = g
    print(f"PTX {name}    best {cfg}: {ms:.3f} ms  {g:.1f} GFLOP/s")
    if best_reg is None or g > best_reg[1]:
        best_reg = (name, g)
print(f"--> register path prefers TT {best_reg[0]} at {best_reg[1]:.1f} GFLOP/s")

# 3) CUDA (NVRTC recompile per config)
cu_src = (HERE / "thread_tiled.cu").read_text()


def cuda_bench(cfg):
    BM, BN, BK, TM, TN = cfg
    if BM % TM or BN % TN:
        return None
    threads = (BM * BN) // (TM * TN)
    if threads % 32 or threads > 1024:
        return None
    if threads % BK or threads % BN:
        return None
    if (BM * BK) % threads or (BK * BN) % threads:
        return None
    strideA, strideB = threads // BK, threads // BN
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
cuda_cfg, cuda_ms, _ = autotune(cuda_space, cuda_bench, label="CUDA")
cuda_gflops = bu.gflops(M, N, K, cuda_ms)
print(f"CUDA       best {cuda_cfg}: {cuda_ms:.3f} ms  {cuda_gflops:.1f} GFLOP/s")

# 4) Triton FP32 + cuBLAS FP32 : fixed references (measured once in reference_fp32.py)
tri_gflops, cublas_gflops = TRITON_FP32, CUBLAS_FP32

# ---------------------------------------------------------------------------
# charts + results
# ---------------------------------------------------------------------------
(HERE / "results.json").write_text(json.dumps({
    "ptx_local": ptx_local, "ptx_regs_by_tt": reg_gflops,
    "cuda": cuda_gflops, "triton": tri_gflops, "cublas_fp32": cublas_gflops,
}, indent=2))

# which TT the register path prefers
plot_bars({f"TT {k}": reg_gflops[k] for k in ("2x2", "4x4", "8x8")},
          out_path=str(HERE / "tt_sweep.png"))

# overall picture, best register TT vs everything else
plot_bars({"PTX (regs)": best_reg[1],
           "CUDA": cuda_gflops, "Triton": tri_gflops, "cuBLAS": cublas_gflops},
          out_path=str(HERE / "thread_tiling_all.png"))

print("wrote tt_sweep.png, thread_tiling_all.png, results.json")
