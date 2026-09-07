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


def _is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check(mapping, spec, prefix=""):
    errors = []
    for key, kind in spec.items():
        label = f"{prefix}{key}"
        if key not in mapping:
            errors.append(f"missing: {label}")
            continue
        value = mapping[key]
        if kind == "num":
            if not _is_num(value):
                errors.append(f"bad type {label}: expected number, got {type(value).__name__}")
        elif kind == "str":
            if not isinstance(value, str):
                errors.append(f"bad type {label}: expected str, got {type(value).__name__}")
        elif kind == "bool":
            if not isinstance(value, bool):
                errors.append(f"bad type {label}: expected bool, got {type(value).__name__}")
        elif kind == "dict":
            if not isinstance(value, dict):
                errors.append(f"bad type {label}: expected dict, got {type(value).__name__}")
        elif kind == "list":
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
    return errors
