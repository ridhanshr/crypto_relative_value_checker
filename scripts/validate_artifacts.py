"""Validate JSON artifacts produced by a checker run."""
import argparse
import json
from pathlib import Path
from crypto_checker.schema import validate_output_schema

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()
root = Path(args.path)
kind_by_name = {"decision.json": "result", "validation.json": "validation",
                "walk_forward.json": "walk_forward", "capacity_curve.json": "capacity",
                "preflight.json": "preflight"}
errors = []
for path in root.rglob("*.json"):
    kind = kind_by_name.get(path.name)
    if not kind:
        continue
    try:
        errors.extend(f"{path}: {error}" for error in validate_output_schema(json.loads(path.read_text()), kind))
    except Exception as exc:
        errors.append(f"{path}: {type(exc).__name__}: {exc}")
if errors:
    raise SystemExit("\n".join(errors))
print("SCHEMA_ARTIFACTS_OK")
