"""Output contracts for Quantara integration: SCHEMA_VERSION + validator.

Every JSON artifact the checker emits carries ``schema_version``. Rules:

* ADDING a field = minor, allowed; the validator ignores unknown keys.
* REMOVING or RE-TYPING a locked field = major; the contract test fails
  until SCHEMA_VERSION is bumped and Quantara adapts.
* ``validate_output_schema(artifact, kind)`` returns a list of violations
  (empty = valid). Quantara should run it on every artifact it ingests:
  a schema-valid file is never a half-written or corrupt one structurally
  (atomic write in io helpers guarantees the bytes; this guarantees the
  shape).
"""

SCHEMA_VERSION = 1

# Versioning contract (frozen with Quantara):
#   v1.0 -> v1.1 : ADD fields (backward-compatible). schema_version stays 1.
#   v1.x -> v2   : RENAME / REMOVE / TYPE-CHANGE / SEMANTIC-BREAK.
#                  schema_version becomes 2. This file is currently v1.1.
SCHEMA_CHANGELOG = [
    "v1.0: decision, validation, walk_forward, capacity, preflight locked",
    "v1.1: envelope kind 'result' + 'analysis' added (additive only); "
    "nullable spec kinds ('str?', 'num?', 'dict?') for explicit nulls",
]

DEPLOYABLE_MEANING = (
    "Lolos validation criteria checker. BUKAN izin live trading / real money. "
    "Promotion to paper/live is Quantara's decision under its own criteria."
)


def _is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check(mapping, spec, prefix=""):
    errors = []
    for key, kind in spec.items():
        # Trailing "?" = nullable: explicit null is contract-valid.
        nullable = kind.endswith("?")
        base = kind[:-1] if nullable else kind
        label = f"{prefix}{key}"
        if key not in mapping:
            errors.append(f"missing: {label}")
            continue
        value = mapping[key]
        if value is None:
            if not nullable:
                errors.append(f"null forbidden: {label}")
            continue
        if base == "num":
            if not _is_num(value):
                errors.append(f"bad type {label}: expected number, got {type(value).__name__}")
        elif base == "str":
            if not isinstance(value, str):
                errors.append(f"bad type {label}: expected str, got {type(value).__name__}")
        elif base == "bool":
            if not isinstance(value, bool):
                errors.append(f"bad type {label}: expected bool, got {type(value).__name__}")
        elif base == "dict":
            if not isinstance(value, dict):
                errors.append(f"bad type {label}: expected dict, got {type(value).__name__}")
        elif base == "list":
            if not isinstance(value, list):
                errors.append(f"bad type {label}: expected list, got {type(value).__name__}")
        else:
            errors.append(f"unknown spec kind for {label}")
    return errors


_LEVEL_SPEC = {"aum": "num", "total_return": "num", "sharpe": "num", "max_drawdown": "num", "capacity_violations": "num", "breach_trades": "num", "max_participation_ratio": "num", "headroom_multiple": "num", "sensible": "bool"}

SCHEMAS = {
    # Locked top-level fields per artifact kind. Optional sections
    # (data_mining, ensemble, ...) are validated only when present.
    "decision": {
        "schema_version": "num",
        "best_signal_walk_forward": "str",
        "best_n_sides": "num",
        "gates": "dict",
        "hard_gates": "list",
        "deployable": "bool",
        "walk_forward": "dict",
        "validation_best_signal": "dict",
        "reality_check": "dict",
    },
    "validation": {
        "schema_version": "num",
        "splits": "dict",
        "performance": "dict",
        "gates": "dict",
        "regimes": "list",
        "continuous_equity": "bool",
    },
    "walk_forward": {
        "schema_version": "num",
        "candidates": "list",
        "n_folds": "num",
        "folds": "list",
        "walk_forward": "dict",
        "gates": "dict",
        "deployable": "bool",
    },
    "capacity": {
        "schema_version": "num",
        "status": "str",
        "levels": "list",
    },
    "preflight": {
        "schema_version": "num",
        "valid": "bool",
        "errors": "list",
        "warnings": "list",
    },
    "analysis": {
        "schema_version": "num",
        "mode": "str",
        "assets": "list",
        "preflight": "dict",
        "walk_forward": "dict",
        "deployable": "bool",
    },
    # v1.1 envelope: the single canonical artifact Quantara reads.
    # Population rules (locked):
    #   SUCCESS+APPROVED/REJECTED: metrics/risk/capacity/data_quality populated, decision set.
    #   FAILED_VALIDATION / CHECKER_ERROR: metrics/capacity null, decision null, errors populated.
    "result": {
        "schema_version": "num",
        "status": "str",
        "decision": "str?",
        "deployable": "bool",
        "deployable_meaning": "str",
        "gates": "dict",
        "metrics": "dict?",
        "capacity": "dict?",
        "risk": "dict",
        "data_quality": "dict",
        "warnings": "list",
        "errors": "list",
        "artifacts": "dict",
    },
    "metrics": {
        "wf_total_return": "num",
        "wf_sharpe": "num",
        "wf_drawdown": "num",
        "oos_total_return": "num",
        "oos_sharpe": "num",
        "oos_drawdown": "num",
        "dsr": "num?",
        "turnover_daily": "num",
    },
}


def validate_output_schema(artifact, kind):
    """Return list of contract violations for ``artifact`` of ``kind``."""
    if kind not in SCHEMAS:
        return [f"unknown artifact kind: {kind}"]
    if not isinstance(artifact, dict):
        return [f"artifact must be a dict, got {type(artifact).__name__}"]
    errors = _check(artifact, SCHEMAS[kind])
    if artifact.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version: expected {SCHEMA_VERSION}, got {artifact.get('schema_version')!r}")
    if kind == "capacity" and isinstance(artifact.get("levels"), list):
        for i, level in enumerate(artifact["levels"]):
            if isinstance(level, dict):
                errors.extend(_check(level, _LEVEL_SPEC, prefix=f"levels[{i}]."))
            else:
                errors.append(f"levels[{i}]: expected dict, got {type(level).__name__}")
    if kind == "result":
        if isinstance(artifact.get("metrics"), dict):
            errors.extend(_check(artifact["metrics"], SCHEMAS["metrics"], prefix="metrics."))
        if isinstance(artifact.get("capacity"), dict) and "max_sensible" in artifact["capacity"]:
            ms = artifact["capacity"]["max_sensible"]
            if ms is not None and not _is_num(ms):
                errors.append(f"bad type capacity.max_sensible: expected number|null, got {type(ms).__name__}")
        if artifact.get("status") not in ("SUCCESS", "FAILED_VALIDATION", "CHECKER_ERROR"):
            errors.append(f"bad status: {artifact.get('status')!r}")
        if artifact.get("decision") not in ("APPROVED", "REJECTED", None):
            errors.append(f"bad decision: {artifact.get('decision')!r}")
    return errors
