import pandas as pd
from crypto_checker.core import CheckerConfig, check_strategy


def test_ranking_exposure_and_costs():
    data = pd.DataFrame([
        ["2026-01-01", "A", 100, 4], ["2026-01-01", "B", 100, 3], ["2026-01-01", "C", 100, 2], ["2026-01-01", "D", 100, 1],
        ["2026-01-02", "A", 110, 1], ["2026-01-02", "B", 100, 2], ["2026-01-02", "C", 100, 3], ["2026-01-02", "D", 90, 4],
    ], columns=["timestamp", "asset", "price", "signal"])
    result = check_strategy(data, CheckerConfig(n_long=1, n_short=1, fee_rate=0.001, slippage_rate=0))
    assert result["ranking"].iloc[0].long_assets == "A"
    assert result["exposure"].iloc[0].gross_exposure == 2
    assert result["pnl"].iloc[1].total_pnl < 0
