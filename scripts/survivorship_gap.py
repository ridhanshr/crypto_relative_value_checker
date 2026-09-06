import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from crypto_checker.lifecycle import measure_survivorship_gap

data = pd.read_csv("data/midcap_2y_daily_2_funded.csv", usecols=["timestamp", "asset"])
manifest = pd.read_csv("data/historical_listing_manifest.csv")
gap = measure_survivorship_gap(data, manifest)
print("window:", gap["window_start"], "->", gap["window_end"], flush=True)
print("manifest in-window:", gap["n_manifest_in_window"], "dataset:", gap["n_dataset"], flush=True)
print("coverage_ratio:", round(gap["coverage_ratio"], 4), flush=True)
print("missing_dead:", gap["n_missing_dead"], flush=True)
dead = gap["missing_dead"]
print("first 60 dead:", ", ".join(dead[:60]), flush=True)
print("missing_alive:", gap["n_missing_alive"], flush=True)
print("DONE", flush=True)
