import numpy as np
import pandas as pd


def newey_west_tstat(values, lags=None):
    x = pd.Series(values).dropna().astype(float).to_numpy()
    n = len(x)
    if n < 2:
        return 0.0
    lags = min(int(lags or max(1, n ** 0.25)), n - 1)
    mean = x.mean()
    variance = ((x - mean) ** 2).mean()
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        variance += 2 * weight * np.mean((x[lag:] - mean) * (x[:-lag] - mean))
    return float(mean / np.sqrt(max(variance, 1e-18) / n))


def block_bootstrap_mean(values, iterations=2000, block_size=5, seed=7):
    x = pd.Series(values).dropna().astype(float).to_numpy()
    if not len(x):
        return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0, "iterations": 0}
    rng = np.random.default_rng(seed)
    blocks = [x[i:i + block_size] for i in range(0, len(x), block_size)]
    samples = [np.concatenate([blocks[i % len(blocks)] for i in rng.integers(0, len(blocks), size=len(blocks))])[:len(x)].mean() for _ in range(iterations)]
    return {"mean": float(x.mean()), "lower_95": float(np.quantile(samples, .025)), "upper_95": float(np.quantile(samples, .975)), "iterations": iterations}


def reality_check(ic_table, returns_by_signal=None):
    table = pd.DataFrame(ic_table)
    if table.empty:
        return {"signals": 0, "best_ic_tstat": 0.0, "bonferroni_alpha": 0.05, "pass": False}
    p = len(table)
    best = float(table["ic_tstat"].abs().max())
    return {"signals": p, "best_ic_tstat": best, "bonferroni_alpha": .05 / p, "multiple_testing_warning": p > 1, "pass": best > 3.0}


def multiple_testing_adjusted_pvalue(tstat, tests):
    if not tests:
        return 1.0
    from math import erfc, sqrt
    return float(min(1.0, tests * erfc(abs(float(tstat)) / sqrt(2))))
