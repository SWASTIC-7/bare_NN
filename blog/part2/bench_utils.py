"""Reusable benchmarking helpers (timing, GFLOP/s, correctness).

Shared across all part-2 sections, so keep it kernel-agnostic.
Deps: cupy, numpy.
"""
import numpy as np


def gflops(M, N, K, ms):
    """A GEMM does 2*M*N*K floating point ops. `ms` is milliseconds per launch."""
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e9


def time_cupy(launch, iters=50, warmup=10):
    """Time a cupy kernel launch (a zero-arg callable) with CUDA events.

    Returns milliseconds per launch (averaged over `iters`).
    """
    import cupy as cp
    for _ in range(warmup):
        launch()
    cp.cuda.runtime.deviceSynchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(iters):
        launch()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / iters


def rel_error(C, ref):
    """Relative Frobenius-norm error between a result and a reference (cupy arrays)."""
    import cupy as cp
    return float(cp.linalg.norm(C - ref) / cp.linalg.norm(ref))


def make_problem(M, N, K, seed=0):
    """Random A (MxK), B (KxN), zero C (MxN) as float32 cupy arrays, plus A@B reference."""
    import cupy as cp
    cp.random.seed(seed)
    A = cp.random.randn(M, K, dtype=cp.float32)
    B = cp.random.randn(K, N, dtype=cp.float32)
    C = cp.zeros((M, N), dtype=cp.float32)
    ref = A @ B
    return A, B, C, ref
