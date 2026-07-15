"""Reference GEMM performance from the vendor libraries (no hand-optimization).

  * cuBLAS  -- via torch.matmul (cuBLAS SGEMM under the hood), FP32 and TF32
  * CUTLASS -- via the nvidia-cutlass Python package (pip install nvidia-cutlass)
Specs: 1024 x 1024 x 1024, NVIDIA GeForce RTX 3050.  Deps: torch, nvidia-cutlass.
"""
import torch


def gflops(M, N, K, ms):
    return 2.0 * M * N * K / (ms * 1e-3) / 1e9


def bench(fn, iters=100, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


M = N = K = 1024
A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)
C = torch.empty(M, N, device="cuda", dtype=torch.float32)

results = {}

# ---- cuBLAS (torch.matmul dispatches to cuBLAS SGEMM) ----
torch.backends.cuda.matmul.allow_tf32 = False
results["cuBLAS FP32"] = gflops(M, N, K, bench(lambda: torch.matmul(A, B, out=C)))
torch.backends.cuda.matmul.allow_tf32 = True
results["cuBLAS TF32"] = gflops(M, N, K, bench(lambda: torch.matmul(A, B, out=C)))

# ---- CUTLASS (Python package) ----
try:
    import cuda                               # cutlass 4.2 checks cuda.__version__; newer bindings moved it
    if not hasattr(cuda, "__version__"):
        cuda.__version__ = "13.3.1"
    import cutlass_cppgen as cutlass          # CUTLASS 4.x Python package
    Cc = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    plan = cutlass.op.Gemm(element=torch.float32, layout=cutlass.LayoutType.RowMajor)
    results["CUTLASS FP32"] = gflops(M, N, K, bench(lambda: plan.run(A, B, Cc, Cc)))
except Exception as ex:
    print(f"[CUTLASS skipped] {type(ex).__name__}: {ex}")

for k, v in results.items():
    print(f"{k:16s} {v:8.1f} GFLOP/s")
