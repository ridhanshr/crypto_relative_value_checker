"""Fase 1 tests: trial registry + DSR v2."""

import numpy as np
import pandas as pd
import pytest

from crypto_checker.trial_registry import TrialRegistry, ExclusionReason
from crypto_checker.dsr import deflated_sharpe_ratio


def test_all_attempts_logged_including_failed():
    reg = TrialRegistry()
    reg.log_trial("good", {"signal": "a"}, 1.2, returns=pd.Series(np.random.default_rng(0).normal(size=100)))
    reg.log_trial("crashed", {"signal": "b"}, None, ExclusionReason.DATA_ERROR)
    reg.log_trial("short", {"signal": "c"}, None, ExclusionReason.INSUFFICIENT_SAMPLE)
    assert reg.raw_n_trials() == 3
    assert len(reg.valid_trials()) == 1
    assert reg.valid_sharpes() == [1.2]


def test_exclusion_requires_structured_reason():
    reg = TrialRegistry()
    with pytest.raises(ValueError, match="exclusion_reason"):
        reg.log_trial("x", {}, None, None)
    with pytest.raises(ValueError, match="ExclusionReason"):
        reg.log_trial("x", {}, None, "just because")
    with pytest.raises(ValueError, match="contradictory"):
        reg.log_trial("x", {}, 1.0, ExclusionReason.DATA_ERROR)


def test_effective_n_trials_excludes_invalid_and_clusters_correlated():
    reg = TrialRegistry()
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(size=200))
    reg.log_trial("a", {"s": "a"}, 1.0, returns=base)
    reg.log_trial("b", {"s": "b"}, 0.9, returns=base * 1.0)  # near-duplicate -> same cluster
    reg.log_trial("c", {"s": "c"}, 0.5, returns=pd.Series(rng.normal(size=200)))
    reg.log_trial("dead", {"s": "d"}, None, ExclusionReason.CONVERGENCE_FAILURE)
    assert reg.raw_n_trials() == 4
    assert reg.effective_n_trials(0.9) == 2  # {a,b} cluster + {c}; dead excluded


def test_dsr_shape_and_adjustment_flag():
    rng = np.random.default_rng(2)
    out = deflated_sharpe_ratio(0.8, 10, pd.Series(rng.normal(size=300)), skill_free_variance=0.04)
    assert set(out) == {"dsr", "sharpe_null_bar", "skewness", "kurtosis", "n_trials_used", "adjustment_applied"}
    assert out["n_trials_used"] == 10 and out["adjustment_applied"] is True
    assert 0.0 <= out["dsr"] <= 1.0
    # Pearson kurtosis of normal ~ 3, not ~ 0.
    assert 2.0 < out["kurtosis"] < 4.0


def test_dsr_lower_with_negative_skew():
    rng = np.random.default_rng(3)
    symmetric = pd.Series(rng.normal(size=500))
    skewed = pd.Series(np.concatenate([rng.normal(size=490), [-6.0] * 10]))
    kw = dict(observed_sharpe=0.5, n_trials=10, skill_free_variance=0.04)
    assert deflated_sharpe_ratio(returns=skewed, **kw)["dsr"] < deflated_sharpe_ratio(returns=symmetric, **kw)["dsr"]


def test_dsr_uses_effective_trials_not_raw():
    rng = np.random.default_rng(4)
    rets = pd.Series(rng.normal(size=500))
    low = deflated_sharpe_ratio(1.0, 5, rets, skill_free_variance=0.04)["dsr"]
    high = deflated_sharpe_ratio(1.0, 200, rets, skill_free_variance=0.04)["dsr"]
    assert high < low


def test_dsr_fails_loudly_on_short_or_degenerate_data():
    with pytest.raises(ValueError, match="at least"):
        deflated_sharpe_ratio(1.0, 5, pd.Series([0.1, 0.2, 0.3]), skill_free_variance=0.04)
    with pytest.raises(ValueError, match="zero-variance"):
        deflated_sharpe_ratio(1.0, 5, pd.Series([0.0] * 500), skill_free_variance=0.04)
    with pytest.raises(ValueError, match="skill_free_variance"):
        deflated_sharpe_ratio(1.0, 50, pd.Series(np.random.default_rng(5).normal(size=500)))


def test_effective_n_never_exceeds_raw_invariant():
    """Regression test for bug where effective_n_trials exceeded raw_n_trials
    due to cross-fold double-counting (clustering across folds duplicated
    correlated trials instead of clustering per-fold then summing)."""
    from crypto_checker.trial_registry import TrialRegistry
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(999)
    reg = TrialRegistry()
    # Create 4 distinct base return series (one per signal)
    bases = {sig: pd.Series(np.random.default_rng(i).normal(size=200)) for i, sig in enumerate(["a", "b", "c", "d"])}
    for fold in range(5):
        for i, signal in enumerate(["a", "b", "c", "d"]):
            trial_id = f"fold{fold}|{signal}"
            # Same params repeated across folds -> should cluster together
            reg.log_trial(f"fold{fold}|{signal}", {"signal": signal}, 1.0,
                         returns=bases[signal] + rng.normal(scale=1e-9, size=200))
    # raw = 20 attempts (5 folds × 4 signals), but only 4 unique configs
    # effective_n should be 4 (one per unique config), not 20
    assert reg.raw_n_trials() == 20
    assert reg.effective_n_trials(0.9) == 4
    # Invariant: effective_n_trials must never exceed raw_n_trials
    assert reg.effective_n_trials(0.9) <= reg.raw_n_trials()
