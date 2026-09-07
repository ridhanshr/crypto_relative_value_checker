"""CPCV on any best config (generalized from the low_vol_14 confirmation).

Fresh-deployment semantics per path (same as WF folds).
Example:
  python scripts/run_cpcv_strategy.py --dataset data/midcap_2y_daily_2_repaired.csv --signal vol_adj_momentum_30 --out reports/cpcv_mom30 --offset 0 --limit 9
  ... repeat offsets ... then --summarize
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import pandas as pd
from crypto_checker.core import CheckerConfig, check_strategy
from crypto_checker.signals import build_signals
from crypto_checker.cpcv import generate_cpcv_splits, label_regime, summarize_cpcv, FoldResult


def norm_equity(pnl):
    eq = pd.Series(pnl["equity"].to_numpy(), dtype=float)
    return eq / eq.iloc[0]


def run_path(train, test, cfg, fold_id, train_idx, test_idx, regime):
    train_res = check_strategy(train, cfg)
    test_res = check_strategy(test, cfg)
    test_pnl = test_res["pnl"]
    test_returns = pd.Series(test_pnl["return"].iloc[1:].to_numpy(),
                             index=pd.to_datetime(test_pnl["timestamp"].iloc[1:], utc=True))
    std = test_returns.std(ddof=1)
    sharpe = float(test_returns.mean() / std * 365.0 ** 0.5) if std else 0.0
    test_equity = norm_equity(test_pnl)
    test_return = float(test_equity.iloc[-1] - 1)
    return {"fold_id": fold_id, "train_idx": [str(t) for t in train_idx],
            "test_idx": [str(t) for t in test_idx], "sharpe_oos": sharpe,
            "test_return": test_return,
            "equity_curve_train": norm_equity(train_res["pnl"]).tolist(),
            "equity_curve_oos": test_equity.tolist(), "regime_label": regime}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--signal", required=True)
    parser.add_argument("--n-long", type=int, default=3)
    parser.add_argument("--n-short", type=int, default=3)
    parser.add_argument("--vol-target", type=float, default=0.20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=45)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths_file = out / "fold_results.jsonl"
    cfg = CheckerConfig(n_long=args.n_long, n_short=args.n_short, fee_rate=0.0004, slippage_rate=0.0005,
                        signal_column=args.signal, vol_target_annual=args.vol_target, rebalance_every=5,
                        require_funding=True, delist_mode="forced_exit")
    if args.summarize:
        results = []
        for line in paths_file.read_text().splitlines():
            d = json.loads(line)
            results.append(FoldResult(
                fold_id=d["fold_id"], train_idx=pd.Index(pd.to_datetime(d["train_idx"], utc=True)),
                test_idx=pd.Index(pd.to_datetime(d["test_idx"], utc=True)), sharpe_oos=d["sharpe_oos"],
                test_return=d["test_return"], equity_curve_train=pd.Series(d["equity_curve_train"]),
                equity_curve_oos=pd.Series(d["equity_curve_oos"]), regime_label=d["regime_label"]))
        summary = summarize_cpcv(results)
        (out / "cpcv_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps(summary, indent=1, default=str), flush=True)
        print("CPCV_SUMMARY_DONE", flush=True)
        return
    data = build_signals(pd.read_csv(args.dataset))
    data = data.dropna(subset=[args.signal]).reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    stamps = data["timestamp"].drop_duplicates().sort_values()
    splits = generate_cpcv_splits(stamps, 10, 2, 0.01, 0.01)
    ref = data.sort_values("timestamp").groupby("timestamp")["price"].last()
    vol = label_regime(ref.pct_change(), method="realized_vol_percentile")
    done = set()
    if paths_file.exists():
        for line in paths_file.read_text().splitlines():
            done.add(json.loads(line)["fold_id"])
    for k in range(args.offset, min(args.offset + args.limit, len(splits))):
        if k in done:
            print(f"fold {k} cached", flush=True)
            continue
        train_idx, test_idx = splits[k]
        train = data[data.timestamp.isin(train_idx)]
        test = data[data.timestamp.isin(test_idx)]
        regime = vol.reindex(test_idx).dropna()
        regime_label = str(regime.mode().iloc[0]) if len(regime) else "unknown"
        print(f"fold {k}/{len(splits)} regime={regime_label} train={len(train)} test={len(test)}", flush=True)
        row = run_path(train, test, cfg, k, train_idx, test_idx, regime_label)
        with open(paths_file, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    print("CHUNK_DONE", flush=True)


if __name__ == "__main__":
    main()
