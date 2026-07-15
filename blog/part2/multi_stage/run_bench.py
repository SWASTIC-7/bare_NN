"""Benchmark for the 'Multi-stage pipeline' section: multi_stage.ptx (3-stage) vs CUDA vs Triton.
Run: python run_bench.py    Deps: cupy, torch, triton, matplotlib, numpy.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from section_bench import run_section

# 3-stage cp.async 8x4 kernel, baked config, 3 shared buffers
run_section(HERE / "multi_stage.ptx", (64, 128, 8, 8, 4, 4, 8), nbuf=3, out_png=HERE / "multi_stage_all.png")
