"""Trial registry: log EVERY backtest attempt, count effective trials.

No code path may skip logging -- including trials that crash or are too
short. Attempts without a computable Sharpe are logged with valid=False and
a STRUCTURED ExclusionReason (enum only, never free text), so exclusion can
never become a cover for cherry-picking. They count in raw_n_trials but are
excluded from variance/clustering math with a documented reason.

effective_n_trials() clusters return series greedily: each valid trial joins
the first cluster whose members all correlate above the threshold with it,
else it opens a new cluster. Trials without stored returns count as
singletons (conservative = separate). Deterministic: trial order is sorted
by trial_id before clustering, so identical inputs give identical counts on
any PYTHONHASHSEED.
"""

from dataclasses import dataclass
from enum import Enum
import pandas as pd


class ExclusionReason(Enum):
    CONVERGENCE_FAILURE = "convergence_failure"
    INSUFFICIENT_SAMPLE = "insufficient_sample"
    DATA_ERROR = "data_error"


@dataclass
class TrialRecord:
    trial_id: str
    params: dict
    sharpe: float | None
    valid: bool = True
    exclusion_reason: ExclusionReason | None = None


class TrialRegistry:
    def __init__(self):
        self.records: list[TrialRecord] = []
        self._returns: dict[str, pd.Series] = {}

    def log_trial(self, trial_id, params, sharpe=None, exclusion_reason=None, returns=None):
        if sharpe is None and exclusion_reason is None:
            raise ValueError(f"Trial {trial_id}: sharpe=None requires a structured exclusion_reason")
        if sharpe is not None and exclusion_reason is not None:
            raise ValueError(f"Trial {trial_id}: sharpe given with exclusion_reason is contradictory")
        if exclusion_reason is not None and not isinstance(exclusion_reason, ExclusionReason):
            raise ValueError(f"Trial {trial_id}: exclusion_reason must be an ExclusionReason, got {exclusion_reason!r}")
        valid = sharpe is not None and pd.notna(sharpe)
        record = TrialRecord(trial_id=str(trial_id), params=dict(params),
                             sharpe=None if not valid else float(sharpe),
                             valid=bool(valid),
                             exclusion_reason=None if valid else exclusion_reason)
        self.records.append(record)
        if returns is not None and valid:
            # Keep the timestamp index: cross-fold correlation aligns on
            # overlapping timestamps, never positionally.
            self._returns[record.trial_id] = pd.Series(returns)
        return record

    def raw_n_trials(self):
        return len(self.records)

    def valid_trials(self):
        return [r for r in self.records if r.valid]

    def valid_sharpes(self):
        return [r.sharpe for r in self.valid_trials()]

    MIN_OVERLAP = 30

    def _aligned_corr(self, a, b):
        """Correlation on inner-joined timestamps (positional only when both
        lack a datetime index and share length). Insufficient overlap or
        degenerate series -> NaN (caller treats as separate clusters)."""
        a, b = pd.Series(a), pd.Series(b)
        a_dt = isinstance(a.index, pd.DatetimeIndex)
        b_dt = isinstance(b.index, pd.DatetimeIndex)
        if a_dt and b_dt:
            joined = pd.concat([a, b], axis=1, join="inner").dropna()
            if len(joined) < self.MIN_OVERLAP:
                return float("nan")
            return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        if len(a) == len(b) and len(a) >= 2:
            return float(pd.Series(a.to_numpy()).corr(pd.Series(b.to_numpy())))
        return float("nan")

    def effective_n_trials(self, corr_threshold=0.9, series=None):
        """Greedy correlation clustering over valid trials with returns.

        ``series`` optionally overrides stored returns (used for per-fold
        diagnostics over equal-length slices). Trials without usable
        returns are singletons. NaN/degenerate correlations never merge.
        """
        pool = self.valid_trials()
        store = dict(self._returns) if series is None else dict(series)
        if series is not None:
            # Per-fold diagnostics scope to that fold's trials only; without
            # this, earlier folds leak in as phantom singletons.
            pool = [r for r in pool if r.trial_id in store]
        usable = [(r.trial_id, store[r.trial_id]) for r in pool
                  if r.trial_id in store and len(pd.Series(store[r.trial_id]).dropna()) >= 2]
        singletons = len(pool) - len(usable)
        clusters: list[list] = []
        by_id = dict(usable)
        for tid in sorted(by_id):
            member = by_id[tid]
            placed = False
            for cluster in clusters:
                corrs = [self._aligned_corr(member, other) for _, other in cluster]
                if corrs and all(pd.notna(c) and c > corr_threshold for c in corrs):
                    cluster.append((tid, member))
                    placed = True
                    break
            if not placed:
                clusters.append([(tid, member)])
        return len(clusters) + singletons
