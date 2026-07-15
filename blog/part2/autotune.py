"""Reusable autotuner: sweep a list of configs, keep the fastest valid one.

Shared across all part-2 sections. It knows nothing about the kernel; the caller
passes a `bench_fn(config)` that returns milliseconds, or None to skip a config
(invalid launch params or wrong numerical result).
"""


def autotune(configs, bench_fn, label="kernel", verbose=True):
    """
    configs  : iterable of config objects (tuples/dicts) to try
    bench_fn : callable(config) -> milliseconds, or None to skip
    returns  : (best_config, best_ms, all_results)  where all_results is a list of
               (config, ms) for every config that ran successfully.
    """
    best = None
    results = []
    for cfg in configs:
        try:
            ms = bench_fn(cfg)
        except Exception as e:                       # a bad launch shouldn't kill the sweep
            ms = None
            if verbose:
                print(f"[{label}] skip {cfg}: {type(e).__name__}: {e}")
        if ms is None:
            continue
        results.append((cfg, ms))
        if best is None or ms < best[1]:
            best = (cfg, ms)
            if verbose:
                print(f"[{label}] new best {cfg}: {ms:.3f} ms")
    if best is None:
        raise RuntimeError(f"[{label}] no valid config found")
    return best[0], best[1], results
