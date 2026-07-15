"""Benchmark for the 'Vectorized register-staged' section (MIO-throttle-free) vs CUDA vs Triton.
Run: python run_bench.py    Deps: cupy, torch, triton, matplotlib, numpy.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from section_bench import run_section

run_section(HERE / "vectorized_register_staged.ptx", (64, 128, 8, 8, 4, 4, 8), nbuf=2,
            out_png=HERE / "vectorized_register_staged_all.png")
