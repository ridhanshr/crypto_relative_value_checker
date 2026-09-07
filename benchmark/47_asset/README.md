# 47-asset funded dataset (happy path / baseline benchmark)

dataset_id: midcap_2y_daily_2_funded
dataset_version: 1
sha256: 53e2db604567de92fbcbf10525a0b36707d67da9c5aae050c03a5df74a65d12d
filename: midcap_2y_daily_2_funded.csv
date_range: 2024-01-01 to 2026-08-31
universe: 47 assets, midcap USDT-M futures + mega-cap reference
source: Binance Vision klines + monthly fundingRate + fapi fallback; repaired (stale volume-zero rows dropped, GAL/OMNI official ratios)
rows: 44911
funding_missing_pct: 0.0
role: baseline benchmark (clean data proves the engine works on the happy path)
