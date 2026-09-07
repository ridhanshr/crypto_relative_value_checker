"""Ensemble pre-check: correlate member OOS returns BEFORE assembling.

Runs on the member test-return series walk_forward already computes per
fold. PROCEED = diverse enough; DROP_REDUNDANT = some pairs overlap;
BLOCK_ENSEMBLE = everything moves together (assembling adds cost without
diversification). Deterministic: sorted pair order, no RNG.
"""

import itertools
import pandas as pd


def compute_pairwise_return_correlation(member_returns):
    """Correlation of ACTUAL OOS return series (not parameter similarity)."""
    frame = pd.DataFrame({name: pd.Series(series).reset_index(drop=True)
                          for name, series in member_returns.items()})
    return frame.corr()


def flag_redundant_members(corr_matrix, threshold):
    redundant = []
    names = list(corr_matrix.columns)
    for a, b in itertools.combinations(sorted(names), 2):
        corr = corr_matrix.loc[a, b]
        if pd.notna(corr) and corr > threshold:
            redundant.append((a, b, float(corr)))
    return redundant


def pre_ensemble_report(candidate_members, config):
    """Decide BEFORE the full ensemble backtest whether assembly is worth it."""
    names = sorted(candidate_members)
    if len(names) < 2:
        return {"max_pairwise_corr": 0.0, "redundant_pairs": [],
                "recommendation": "PROCEED", "note": "single member"}
    corr = compute_pairwise_return_correlation({n: candidate_members[n] for n in names})
    pairs = flag_redundant_members(corr, config.ensemble_max_pairwise_corr)
    off_diag = [corr.loc[a, b] for a, b in itertools.combinations(names, 2) if pd.notna(corr.loc[a, b])]
    max_corr = float(max(off_diag)) if off_diag else 0.0
    if pairs and len(pairs) >= len(names) * (len(names) - 1) // 2:
        recommendation = "BLOCK_ENSEMBLE"
    elif pairs:
        recommendation = "DROP_REDUNDANT"
    else:
        recommendation = "PROCEED"
    return {"max_pairwise_corr": max_corr,
            "redundant_pairs": [[a, b, c] for a, b, c in pairs],
            "recommendation": recommendation}
