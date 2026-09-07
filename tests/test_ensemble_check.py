"""Fase 3 tests: ensemble correlation pre-check."""

import numpy as np
import pandas as pd

from crypto_checker.ensemble_check import (
    compute_pairwise_return_correlation,
    flag_redundant_members,
    pre_ensemble_report,
)


class Cfg:
    ensemble_max_pairwise_corr = 0.7


def test_flags_high_correlation_pairs():
    rng = np.random.default_rng(11)
    base = pd.Series(rng.normal(size=200))
    members = {"a": base, "b": base * 1.0 + rng.normal(scale=1e-9, size=200), "c": pd.Series(rng.normal(size=200))}
    corr = compute_pairwise_return_correlation(members)
    pairs = flag_redundant_members(corr, 0.7)
    names = {(a, b) for a, b, _ in pairs}
    assert ("a", "b") in names
    assert not any("c" in (a, b) for a, b, _ in pairs)


def test_recommendation_block_when_all_members_correlated():
    rng = np.random.default_rng(12)
    base = pd.Series(rng.normal(size=200))
    members = {f"m{i}": base + rng.normal(scale=1e-9, size=200) for i in range(3)}
    report = pre_ensemble_report(members, Cfg())
    assert report["recommendation"] == "BLOCK_ENSEMBLE"
    assert report["max_pairwise_corr"] > 0.99


def test_recommendation_proceed_when_diverse():
    rng = np.random.default_rng(13)
    members = {f"m{i}": pd.Series(rng.normal(size=200)) for i in range(3)}
    report = pre_ensemble_report(members, Cfg())
    assert report["recommendation"] == "PROCEED"
    assert report["redundant_pairs"] == []


def test_single_member_proceeds():
    report = pre_ensemble_report({"only": pd.Series([0.1, -0.2, 0.3])}, Cfg())
    assert report["recommendation"] == "PROCEED"
