"""Minimal launcher for Nsight Compute (ncu) profiling of a PTX kernel.

ncu can't profile a bare .ptx -- it profiles a running kernel. This is the tiny
host program that loads the PTX, makes matrices, and launches once.

Usage:
    python profile_ptx.py <ptx_path> <BT_M> <BT_N> <BT_K> <WT_X> <WT_Y> <TT_X> <TT_Y> <nbuf>
      nbuf = 1 for single-buffer kernels, 2 for double-buffered ones.

Then profile it, e.g.:
    ncu --set full -c 1 -k thread_tiled_ptx python profile_ptx.py \
        double_buffering/compiled_buffering_8x4.ptx 64 128 8 8 4 4 8 2

Deps: cupy, numpy.
"""
import sys
import numpy as np
import cupy as cp

ptx = sys.argv[1]
BT_M, BT_N, BT_K, WT_X, WT_Y, TT_X, TT_Y, nbuf = map(int, sys.argv[2:10])

M = N = K = 1024
A = cp.asarray(np.random.randn(M, K).astype(np.float32))
B = cp.asarray(np.random.randn(K, N).astype(np.float32))
C = cp.zeros((M, N), dtype=cp.float32)

fn = cp.RawModule(path=ptx).get_function("thread_tiled_ptx")
threads = (BT_N // TT_X) * (BT_M // TT_Y)
grid = ((N + BT_N - 1) // BT_N, (M + BT_M - 1) // BT_M, 1)
block = (threads, 1, 1)
smem = nbuf * (BT_K * (BT_M + 4) + BT_K * BT_N) * 4   # +4 padding; nbuf buffers
args = (C, A, B, np.int32(M), np.int32(N), np.int32(K),
        np.int32(BT_M), np.int32(BT_N), np.int32(BT_K),
        np.int32(WT_X), np.int32(WT_Y), np.int32(TT_X), np.int32(TT_Y))

fn(grid, block, args, shared_mem=smem)   # the single launch ncu profiles
cp.cuda.runtime.deviceSynchronize()
print("launched", ptx)
