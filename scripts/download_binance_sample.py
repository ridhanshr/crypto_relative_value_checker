from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crypto_checker.binance_vision import download_binance_daily

download_binance_daily(["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"], "2026-08-01", "2026-09-01", "data/binance_real_case.csv")
print("Wrote data/binance_real_case.csv")
