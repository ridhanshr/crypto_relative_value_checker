"""Thin CLI over the official API: python -m crypto_checker.validate.

Exit codes (locked contract):
  0 = SUCCESS (decision APPROVED or REJECTED -- REJECTED is a verdict, not an error),
  2 = FAILED_VALIDATION (decision.json still written),
  1 = CHECKER_ERROR (decision.json still attempted).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crypto_checker.api import validate_csv, STATUS_FOR_EXIT


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate a strategy dataset (official Quantara entry point)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/validation_run")
    parser.add_argument("--n-long", type=int, default=3)
    parser.add_argument("--n-short", type=int, default=3)
    parser.add_argument("--min-train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=120)
    parser.add_argument("--ensemble-top-k", type=int, default=3)
    parser.add_argument("--listing-manifest", default="")
    args = parser.parse_args(argv)
    envelope = validate_csv(
        args.input, output_dir=args.output, n_long=args.n_long, n_short=args.n_short,
        min_train_days=args.min_train_days, test_days=args.test_days,
        ensemble_top_k=args.ensemble_top_k,
        listing_manifest=args.listing_manifest or None,
    )
    print(json.dumps({"status": envelope["status"], "decision": envelope["decision"],
                      "deployable": envelope["deployable"], "errors": envelope["errors"]}, indent=2))
    return STATUS_FOR_EXIT.get(envelope["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
