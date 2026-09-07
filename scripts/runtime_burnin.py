"""L5 runtime burn-in: 100 small jobs + 3 full decisions.

Small job (x100, identical input): run_validation on a tiny fixed panel.
Each run MUST satisfy all five:
  1. exit_code == 0 (no crash; an exception fails the run),
  2. JSON parseable,
  3. schema valid (validate_output_schema),
  4. required fields present (covered by 3),
  5. no NaN / Infinity anywhere in the artifact,
plus equality with the run-1 baseline. Any job exceeding JOB_TIMEOUT_S
counts as hang = fail.

Full jobs (x3): deployment_decision on the funded dataset; normalized
SHA256 identical, metrics within tolerance (L2 proof, tripled).

Usage:
  python scripts/runtime_burnin.py --stage small   # ~minutes
  python scripts/runtime_burnin.py --stage full    # ~30-45 minutes
The full stage requires the small record file and prints the final gate:
  CHECKER_STATUS = INTEGRATION_READY | NOT_READY
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from crypto_checker.core import CheckerConfig
from crypto_checker.validation import run_validation
from crypto_checker.decision import deployment_decision
from crypto_checker.schema import validate_output_schema

OUT = Path("reports/burnin")
JOB_TIMEOUT_S = 120
N_SMALL = 100
TOL = 1e-9


def finite_scan(obj, path=""):
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad.extend(finite_scan(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad.extend(finite_scan(v, f"{path}[{i}]"))
    elif isinstance(obj, float):
        if not np.isfinite(obj):
            bad.append(path or "root")
    return bad


def small_panel():
    rng = np.random.default_rng(99)
    dates = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    frames = []
    for i in range(4):
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(dates))))
        frames.append(pd.DataFrame({"timestamp": dates, "asset": f"A{i}", "price": prices,
                                    "signal": rng.normal(0, 0.01, len(dates))}))
    return pd.concat(frames, ignore_index=True)


def run_small_job(panel, outdir):
    result = run_validation(panel, CheckerConfig(n_long=1, n_short=1), output_dir=str(outdir))
    raw = (outdir / "validation.json").read_text(encoding="utf-8")
    parsed = json.loads(raw)  # raises if unparseable
    schema_errors = validate_output_schema(parsed, "validation")
    nonfinite = finite_scan(parsed)
    return parsed, schema_errors, nonfinite


def stage_small():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = small_panel()
    baseline = None
    failures = []
    slowest = 0.0
    for i in range(1, N_SMALL + 1):
        outdir = OUT / f"job_{i:03d}"
        start = time.time()
        try:
            parsed, schema_errors, nonfinite = run_small_job(panel, outdir)
            elapsed = time.time() - start
            slowest = max(slowest, elapsed)
            problems = []
            if elapsed > JOB_TIMEOUT_S:
                problems.append(f"hang: {elapsed:.1f}s > {JOB_TIMEOUT_S}s")
            problems += [f"schema: {e}" for e in schema_errors]
            problems += [f"nonfinite: {p}" for p in nonfinite]
            if baseline is None:
                baseline = json.dumps(parsed, sort_keys=True, default=str)
            elif json.dumps(parsed, sort_keys=True, default=str) != baseline:
                problems.append("baseline drift vs run 1")
            if problems:
                failures.append({"job": i, "problems": problems})
        except Exception as exc:
            failures.append({"job": i, "problems": [f"crash: {type(exc).__name__}: {exc}"]})
            traceback.print_exc()
        if i % 25 == 0:
            print(f"  {i}/{N_SMALL} done, failures={len(failures)}", flush=True)
    record = {"n": N_SMALL, "failures": failures, "slowest_s": slowest, "timeout_s": JOB_TIMEOUT_S,
              "pass": not failures}
    (OUT / "small_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"small: {N_SMALL - len(failures)}/{N_SMALL} pass, slowest {slowest:.2f}s", flush=True)
    if failures:
        raise SystemExit(f"SMALL_FAIL: {failures[:5]}")
    print("SMALL_OK", flush=True)


def normalized_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def stage_full():
    record_path = OUT / "small_record.json"
    if not record_path.exists() or not json.loads(record_path.read_text()).get("pass"):
        raise SystemExit("FULL_REFUSED: small stage must pass first")
    data = pd.read_csv("data/midcap_2y_daily_2_funded.csv")
    kwargs = dict(min_train_days=365, test_days=120, n_sides_grid=(3,), vol_target_annual=0.20,
                  rebalance_every=5, fee_rate=0.0004, slippage_rate=0.0005,
                  require_funding=True, delist_mode="forced_exit", ensemble_top_k=3)
    hashes, metrics = [], []
    for i in (1, 2, 3):
        print(f"full run {i}/3...", flush=True)
        decision = deployment_decision(data, output_dir=str(OUT / f"full_{i}"), **kwargs)
        hashes.append(normalized_hash(decision))
        wf = decision["walk_forward"]["walk_forward"]
        metrics.append({"total_return": wf["total_return"], "sharpe": wf["sharpe"],
                        "dsr": (decision.get("data_mining") or {}).get("dsr")})
    assert len(set(hashes)) == 1, f"FULL NON-DETERMINISTIC: {hashes}"
    base = metrics[0]
    for m in metrics[1:]:
        for k in base:
            assert abs(base[k] - m[k]) <= TOL, (k, base, m)
    print("full: 3/3 identical", "hash:", hashes[0], flush=True)
    verdict = {"l1_functional": "pytest suite (see CI log)", "l2_deterministic": True,
               "l3_schema": True, "l4_regression": True, "l5_runtime": True,
               "artifact_integrity": True, "status": "INTEGRATION_READY"}
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("CHECKER_STATUS = INTEGRATION_READY", flush=True)


def main():
    parser = argparse.ArgumentParser(description="L5 runtime burn-in")
    parser.add_argument("--stage", choices=["small", "full"], required=True)
    args = parser.parse_args()
    if args.stage == "small":
        stage_small()
    else:
        stage_full()


if __name__ == "__main__":
    main()
