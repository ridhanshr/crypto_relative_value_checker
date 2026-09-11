import pytest

from crypto_checker.core import CheckerConfig
from crypto_checker.data_quality import DataQualityReport, validate_data_quality


def test_flags_when_unexplained_halts_exceed_threshold():
    report = DataQualityReport(3, 2, 1, 40, 2, 71, 900)
    result = validate_data_quality(report, CheckerConfig(data_quality_max_unexplained_halts=0))
    assert result["status"] == "FLAG_REVIEW"
    assert result["dead_asset_days"] == 40
    assert result["dead_asset_count"] == 2


def test_report_exposes_both_dead_asset_units_separately():
    report = DataQualityReport(1, 0, 1, 123, 4, 71, 900)
    result = validate_data_quality(report, CheckerConfig())
    assert result["status"] == "OK"
    assert result["dead_asset_days"] == 123
    assert result["dead_asset_count"] == 4
    assert "dead" not in result


def test_report_requires_structured_report():
    with pytest.raises(TypeError, match="DataQualityReport"):
        validate_data_quality({}, CheckerConfig())
