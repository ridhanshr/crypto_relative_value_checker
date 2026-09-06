import math
from statistics import NormalDist
import numpy as np
import pandas as pd


EULER_GAMMA = 0.5772156649015329


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


def deflated_sharpe_ratio(observed_sr, n_trials, skew, kurtosis, n_obs, trials_variance, benchmark_sr=0.0):
    """Bailey & de Prado (2014) Deflated Sharpe Ratio.

    Corrects the observed (annualized) Sharpe for selection bias over
    ``n_trials`` tried configurations: DSR = Prob[true SR > benchmark]
    accounting for the expected maximum Sharpe under the null.

    * ``observed_sr``: annualized Sharpe of the SELECTED strategy.
    * ``trials_variance``: variance of the (annualized) Sharpes across all
      tried configurations -- the wider the search, the higher the bar.
    * ``skew``: sample skewness of the selected strategy returns.
    * ``kurtosis``: PEARSON kurtosis (normal == 3) of the selected returns.
    * ``n_obs``: number of return observations.
    * With fewer than 2 trials there is no selection bias and DSR reduces
      to the Probabilistic Sharpe Ratio against ``benchmark_sr``.
    """
    obs = float(observed_sr) - float(benchmark_sr)
    n = int(n_trials)
    T = int(n_obs)
    result = {"n_trials": n, "n_obs": T, "observed_sr": float(observed_sr), "benchmark_sr": float(benchmark_sr)}
    if T < 2:
        return {**result, "expected_sharpe_null": 0.0, "dsr": 0.0, "note": "insufficient_observations"}
    var = float(trials_variance) if trials_variance is not None else 0.0
    if n < 2 or not math.isfinite(var) or var <= 0:
        expected_null = 0.0
    else:
        nd = NormalDist()
        expected_null = math.sqrt(var) * ((1 - EULER_GAMMA) * nd.inv_cdf(1 - 1 / n) + EULER_GAMMA * nd.inv_cdf(1 - 1 / (n * math.e)))
    denom_sq = 1 - float(skew) * obs + (float(kurtosis) - 1) / 4 * obs ** 2
    if not math.isfinite(denom_sq) or denom_sq <= 0:
        return {**result, "expected_sharpe_null": float(expected_null), "dsr": 0.0, "note": "degenerate_return_distribution"}
    stat = (obs - expected_null) * math.sqrt(T - 1) / math.sqrt(denom_sq)
    return {**result, "expected_sharpe_null": float(expected_null), "dsr": float(NormalDist().cdf(stat))}
