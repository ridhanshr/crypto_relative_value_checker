"""Explicit data-quality units for v2 gate decisions.

Never expose ambiguous ``dead`` counts. Reports contain both unique dead
assets and aggregate missing asset-days. Halt counts describe blocks, while
halt days remain available in ``details`` for audit.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DataQualityReport:
    halt_count: int
    halt_unexplained_count: int
    halt_tolerated_count: int
    dead_asset_days: int
    dead_asset_count: int
    universe_size: int
    lookback_days: int

    def to_dict(self):
        return asdict(self)


def validate_data_quality(report, config, details=None):
    """Return OK/FLAG_REVIEW using configured unexplained-halt threshold."""
    if not isinstance(report, DataQualityReport):
        raise TypeError("report must be DataQualityReport")
    reasons = []
    if report.halt_unexplained_count > config.data_quality_max_unexplained_halts:
        reasons.append(
            f"Unexplained halts {report.halt_unexplained_count} > threshold "
            f"{config.data_quality_max_unexplained_halts}"
        )
    result = {"status": "FLAG_REVIEW" if reasons else "OK", "reasons": reasons,
              **report.to_dict()}
    if details is not None:
        result["details"] = dict(details)
    return result


def report_from_preflight(preflight, config):
    """Adapt existing preflight output into explicit v2 quality units."""
    gaps = preflight.get("gap_classification", {}) or {}
    quality = DataQualityReport(
        halt_count=int(gaps.get("halt_blocks", 0)),
        halt_unexplained_count=int(gaps.get("unexplained_blocks", 0)),
        halt_tolerated_count=int(gaps.get("tolerated_blocks", 0)),
        # Historical manifest has asset-level missing counts. Exact missing
        # asset-days require the manifest overlap calculation; keep this
        # explicit rather than relabeling asset count as days.
        dead_asset_days=int((preflight.get("survivorship_gap", {}) or {}).get("missing_dead_asset_days", 0)),
        dead_asset_count=int((preflight.get("survivorship_gap", {}) or {}).get("n_missing_dead", 0)),
        universe_size=int(preflight.get("assets", 0)),
        lookback_days=int(preflight.get("periods", 0)),
    )
    return validate_data_quality(quality, config, details={
        "migration_halt_days": int(gaps.get("migration_halt_days", 0)),
        "unexplained_interior_days": int(gaps.get("unexplained_interior_days", 0)),
    })
