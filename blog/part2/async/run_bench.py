"""Benchmark for the 'Asynchronous copy (cp.async)' section: async.ptx vs CUDA vs Triton.
Run: python run_bench.py    Deps: cupy, torch, triton, matplotlib, numpy.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from section_bench import run_section

# cp.async 8x4 kernel, baked config (BT_M,BT_N,BT_K,WT_X,WT_Y,TT_X,TT_Y), 2 shared buffers
run_section(HERE / "async.ptx", (64, 128, 8, 8, 4, 4, 8), nbuf=2, out_png=HERE / "async_all.png")
