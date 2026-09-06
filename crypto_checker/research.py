from pathlib import Path
import json
import numpy as np
import pandas as pd
from .signals import available_candidates, build_signals
from .reality_check import reality_check, block_bootstrap_mean, newey_west_tstat, multiple_testing_adjusted_pvalue


def _spearman(a, b):
    if len(a) < 3 or a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return np.nan
    ra, rb = a.rank(), b.rank()
    return float(np.corrcoef(ra, rb)[0, 1])


def rank_ic_analysis(frame, candidates, output_dir=None):
    sorted_frame = frame.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    sorted_frame["next_return"] = sorted_frame.groupby("asset")["asset_return"].shift(-1)
    rows = []
    for cand in candidates:
        if cand not in sorted_frame.columns:
            continue
        daily = sorted_frame.dropna(subset=[cand, "next_return"]).groupby("timestamp").apply(lambda g: _spearman(g[cand], g["next_return"]), include_groups=False)
        daily = daily.dropna()
        if len(daily) < 30:
            rows.append({"signal": cand, "ic_mean": np.nan, "ic_std": np.nan, "ic_tstat": np.nan, "icir": np.nan, "ic_days": int(len(daily))})
            continue
        mean, std = float(daily.mean()), float(daily.std(ddof=1))
        tstat = mean / std * (len(daily) ** 0.5) if std else np.nan
        rows.append({"signal": cand, "ic_mean": mean, "ic_std": std, "ic_tstat": float(tstat), "nw_tstat": newey_west_tstat(daily), "bootstrap_lower_95": block_bootstrap_mean(daily)["lower_95"], "bootstrap_upper_95": block_bootstrap_mean(daily)["upper_95"], "adjusted_pvalue": multiple_testing_adjusted_pvalue(tstat, len(candidates)), "icir": float(mean / std) if std else np.nan, "ic_days": int(len(daily))})
    result = pd.DataFrame(rows).sort_values("ic_tstat", ascending=False, key=lambda s: s.abs() if s.dtype != object else s).reset_index(drop=True)
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        result.to_csv(path / "factor_ic.csv", index=False)
    return result


def quantile_returns(frame, candidates, quantiles=5, output_dir=None):
    sorted_frame = frame.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    sorted_frame["next_return"] = sorted_frame.groupby("asset")["asset_return"].shift(-1)
    rows = []
    for cand in candidates:
        if cand not in sorted_frame.columns:
            continue
        sub = sorted_frame.dropna(subset=[cand, "next_return"]).copy()
        sub["bucket"] = sub.groupby("timestamp")[cand].transform(lambda x: pd.qcut(x, quantiles, labels=False, duplicates="drop"))
        stats = sub.groupby("bucket")["next_return"].agg(["mean", "count"])
        for bucket, row in stats.iterrows():
            rows.append({"signal": cand, "quantile": int(bucket), "mean_next_return": float(row["mean"]), "days": int(row["count"])})
    result = pd.DataFrame(rows)
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        result.to_csv(path / "quantile_returns.csv", index=False)
    return result


def run_research(data, output_dir="reports/research", min_ic_tstat=2.0):
    built = build_signals(data)
    candidates = available_candidates(built)
    ic = rank_ic_analysis(built, candidates, output_dir=output_dir)
    quants = quantile_returns(built, candidates, output_dir=output_dir)
    ic_gate = {row["signal"]: bool(abs(row["ic_tstat"]) >= min_ic_tstat) if not np.isnan(row["ic_tstat"]) else False for _, row in ic.iterrows()}
    records = ic.to_dict("records")
    summary = {"candidates": candidates, "ic_table": records, "ic_gate": ic_gate, "ic_pass_count": int(sum(ic_gate.values())), "reality_check": reality_check(records), "bootstrap_note": "Use block_bootstrap_mean on daily IC series for candidate-specific confidence intervals"}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "research_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary, built
