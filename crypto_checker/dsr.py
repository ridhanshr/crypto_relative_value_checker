"""Deflated Sharpe Ratio v2 (Bailey & Lopez de Prado 2014).

Differences from reality_check.deflated_sharpe_ratio (kept for compat):
* input is the OOS return SERIES (skew/kurtosis always computed, never
  passed in) plus the EFFECTIVE trial count from TrialRegistry;
* fail loudly when skew/kurtosis are uncomputable (short/degenerate data)
  instead of returning a quiet 0.0;
* report shape matches the v2 contract (n_trials_used, raw context carried
  by the caller, adjustment flag).

Statistical notes (locked, do not "simplify"):
* kurtosis reported is PEARSON (normal == 3): pandas .kurt() is excess, so
  +3. The PSR denominator needs Pearson gamma_4. The spec example
  (kurtosis 3.1) confirms Pearson.
* skill_free_variance is V, the variance of trial Sharpes under no skill.
  If None and n_trials > 1 the null bar is unidentified -> fail loudly
  (a guessed V silently sets the bar).
"""

import math
from statistics import NormalDist
import numpy as np
import pandas as pd

from .reality_check import EULER_GAMMA

MIN_OBS_FOR_MOMENTS = 4


def deflated_sharpe_ratio(observed_sharpe, n_trials, returns, skill_free_variance=None, use_adjustment=True):
    series = pd.Series(returns).dropna().astype(float)
    n_obs = int(len(series))
    if n_obs < MIN_OBS_FOR_MOMENTS:
        raise ValueError(f"DSR needs at least {MIN_OBS_FOR_MOMENTS} observations for skew/kurtosis, got {n_obs}")
    skew = float(series.skew())
    kurt_pearson = float(series.kurt() + 3)
    if not math.isfinite(skew) or not math.isfinite(kurt_pearson):
        raise ValueError("DSR skew/kurtosis uncomputable (degenerate returns); refusing normal-formula fallback")
    if series.std(ddof=1) == 0 and float(observed_sharpe) != 0:
        raise ValueError("DSR contradictory input: zero-variance returns with nonzero observed Sharpe")
    n = int(n_trials)
    if n < 1:
        raise ValueError("DSR needs n_trials >= 1")
    obs = float(observed_sharpe)
    if n < 2 or skill_free_variance is None:
        if n >= 2:
            raise ValueError("DSR with n_trials > 1 needs skill_free_variance (trials Sharpe variance); refusing guessed bar")
        expected_null = 0.0
    else:
        var = float(skill_free_variance)
        if not math.isfinite(var) or var < 0:
            raise ValueError("skill_free_variance must be finite and non-negative")
        if var == 0:
            expected_null = 0.0
        else:
            nd = NormalDist()
            expected_null = math.sqrt(var) * ((1 - EULER_GAMMA) * nd.inv_cdf(1 - 1 / n) + EULER_GAMMA * nd.inv_cdf(1 - 1 / (n * math.e)))
    if use_adjustment:
        denom_sq = 1 - skew * obs + (kurt_pearson - 1) / 4 * obs ** 2
    else:
        denom_sq = 1.0
    if not math.isfinite(denom_sq) or denom_sq <= 0:
        raise ValueError("DSR denominator degenerate for these moments; refusing output")
    stat = (obs - expected_null) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
    return {"dsr": float(NormalDist().cdf(stat)), "sharpe_null_bar": float(expected_null),
            "skewness": skew, "kurtosis": kurt_pearson, "n_trials_used": n,
            "adjustment_applied": bool(use_adjustment)}
