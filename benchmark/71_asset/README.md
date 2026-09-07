# 71-asset dataset (dirty-data / stress benchmark)

dataset_id: midcap_2y_daily_2_repaired
dataset_version: 1
sha256: 8b07528ec9d3756526d810c62933fa4f20f7ab0466c12196e5189031719365c5
filename: midcap_2y_daily_2_repaired.csv
date_range: 2024-01-02 to 2026-08-31
universe: 71 raw symbols (-> 71 canonical assets after repair; NOM canonical excluded: unverifiable -30.2% post-factor seam)
source: Binance Vision klines + funding; repaired via scripts/repair_71.py (relabel 211 G + 531 NOM pre-cutover rows, drop 731 stale zero-volume rows, exclude NOM canonical; see data/midcap_2y_daily_2_repair_manifest.json)
rows: 67220
raw_source: data/midcap_2y_daily_2.csv (DO NOT use raw directly: 668 NOM duplicates + mislabeled migration history; strict preflight refuses it by design)
role: stress benchmark (dirty data proves the engine detects, refuses, and verifies repair)
