"""Reusable per-section benchmark: a folder's PTX (fixed baked config) vs best CUDA vs Triton.

Used by the run_bench.py in the async / multi_stage / vectorized_register_staged /
hand_scheduled_ilp folders, whose PTX kernels have their tile dims baked in.
CUDA reference = buffering.cu (autotuned) ; Triton = thread_tiled_triton.py (its own autotune).
Deps: cupy, torch, triton, matplotlib, numpy.
"""
import sys
import json
from pathlib import Path

import numpy as np
import cupy as cp
import bench_utils as bu
from autotune import autotune
from plot_benchmark import plot_bars


def _cuda_best(cu_src, M, N, K, A, B, C, ref):
    def bench(cfg):
        BM, BN, BK, TM, TN = cfg
        if BK % 4 or BN % 4 or TM % 4 or TN % 4 or BM % TM or BN % TN:
            return None
        threads = (BM * BN) // (TM * TN)
        if threads % 32 or threads > 1024:
            return None
        if threads < BM * BK // 4 or threads < BK * BN // 4:
            return None
        if 2 * (BK * (BM + 4) + BK * BN) * 4 > 48 * 1024:
            return None
        opts = (f"-DBM={BM}", f"-DBN={BN}", f"-DBK={BK}", f"-DTM={TM}", f"-DTN={TN}")
        fn = cp.RawModule(code=cu_src, backend="nvrtc", options=opts).get_function("thread_tiled_matmul")
        grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
        block = (threads, 1, 1)
        args = (np.int32(M), np.int32(N), np.int32(K), A, B, C)
        C.fill(0)
        fn(grid, block, args)
        cp.cuda.runtime.deviceSynchronize()
        if bu.rel_error(C, ref) > 1e-3:
            return None
        return bu.time_cupy(lambda: fn(grid, block, args))
    space = [(bm, bn, bk, tm, tn)
             for bm in (64, 128) for bn in (64, 128) for bk in (8, 16)
             for tm in (4, 8) for tn in (4, 8)]
    _, ms, _ = autotune(space, bench, label="CUDA", verbose=False)
    return bu.gflops(M, N, K, ms)


def run_section(ptx_path, cfg, nbuf, out_png):
    """cfg = (BT_M, BT_N, BT_K, WT_X, WT_Y, TT_X, TT_Y) baked into the PTX. nbuf = shared buffers."""
    ptx_path = Path(ptx_path)
    HERE = ptx_path.parent
    sys.path.insert(0, str(HERE))     # so thread_tiled_triton is importable from the folder
    M = N = K = 1024
    A, B, C, ref = bu.make_problem(M, N, K)

    # --- PTX at its fixed baked config ---
    BT_M, BT_N, BT_K, WT_X, WT_Y, TT_X, TT_Y = cfg
    threads = (BT_N // TT_X) * (BT_M // TT_Y)
    grid = ((N + BT_N - 1) // BT_N, (M + BT_M - 1) // BT_M, 1)
    block = (threads, 1, 1)
    smem = nbuf * (BT_K * (BT_M + 4) + BT_K * BT_N) * 4
    args = (C, A, B, np.int32(M), np.int32(N), np.int32(K), *[np.int32(x) for x in cfg])
    fn = cp.RawModule(path=str(ptx_path)).get_function("thread_tiled_ptx")
    C.fill(0)
    fn(grid, block, args, shared_mem=smem)
    cp.cuda.runtime.deviceSynchronize()
    assert bu.rel_error(C, ref) < 1e-3, "PTX produced wrong result"
    ptx_g = bu.gflops(M, N, K, bu.time_cupy(lambda: fn(grid, block, args, shared_mem=smem)))

    # --- CUDA reference (buffering.cu, autotuned) ---
    cuda_g = _cuda_best((HERE / "buffering.cu").read_text(), M, N, K, A, B, C, ref)

    # --- Triton ---
    import torch
    import triton
    from thread_tiled_triton import triton_matmul
    At = torch.from_dlpack(A)
    Bt = torch.from_dlpack(B)
    tri_g = bu.gflops(M, N, K, triton.testing.do_bench(lambda: triton_matmul(At, Bt)))

    plot_bars({"CUDA": cuda_g, "PTX": ptx_g, "Triton": tri_g}, out_path=str(out_png))
    (HERE / "results.json").write_text(json.dumps(
        {"ptx": ptx_g, "cuda": cuda_g, "triton": tri_g}, indent=2))
    print(f"{HERE.name:26s} PTX {ptx_g:7.0f}  CUDA {cuda_g:7.0f}  Triton {tri_g:7.0f}  -> {Path(out_png).name}")
