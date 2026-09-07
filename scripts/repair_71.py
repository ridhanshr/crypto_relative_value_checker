"""Repair file 71-aset (midcap_2y_daily_2.csv) -> midcap_2y_daily_2_repaired.csv.

Thin wrapper over crypto_checker.repair; all rules live in the library and
are regression-tested. Cutoffs = first real print of the target,
cross-checked against data/historical_listing_manifest.csv
(G 2024-08-15, NOM 2025-10-01).

NOM canonical is EXCLUDED (not repaired): post-factor seam -30.2% exceeds
the 20% discontinuity guard after the official 75x ratio, and the disorderly
pre-cutover halt (OMNI frozen at 4.274, zero volume 6+ days) makes
factor-vs-repricing unverifiable. Fail closed, documented in the manifest.

Writes data/midcap_2y_daily_2_repaired.csv + *_repair_manifest.json.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from crypto_checker.repair import repair_panel

SRC = Path("data/midcap_2y_daily_2.csv")
DST = Path("data/midcap_2y_daily_2_repaired.csv")
MANIFEST = Path("data/midcap_2y_daily_2_repair_manifest.json")

CUTOVERS = {
    "GUSDT": ("GALUSDT", "2024-08-15"),
    "NOMUSDT": ("OMNIUSDT", "2025-10-01"),
}

data = pd.read_csv(SRC)
frame, log = repair_panel(data, CUTOVERS, drop_stale=True)

# Seam audit on repaired canonicals (adjusted old vs first new).
frame["_ts"] = pd.to_datetime(frame["timestamp"], utc=True)
seams = {}
for old, new, ratio in (("GALUSDT", "GUSDT", 60.0), ("OMNIUSDT", "NOMUSDT", 75.0)):
    o = frame[frame["asset"].astype(str).str.upper() == old].sort_values("_ts")
    n = frame[frame["asset"].astype(str).str.upper() == new].sort_values("_ts")
    if len(o) and len(n):
        seams[f"{old}->{new}"] = float(n.iloc[0]["price"]) / (float(o.iloc[-1]["price"]) / ratio) - 1
log["seams"] = seams

nom_seam = seams.get("OMNIUSDT->NOMUSDT", 0.0)
if abs(nom_seam) > 0.20:
    nom_mask = frame["asset"].astype(str).str.upper().isin(["OMNIUSDT", "NOMUSDT"])
    log["excluded_NOM_canonical"] = {"rows": int(nom_mask.sum()), "seam": nom_seam,
                                     "reason": "post-factor seam exceeds 20% guard; disorderly pre-cutover halt makes factor-vs-repricing unverifiable"}
    frame = frame.loc[~nom_mask].reset_index(drop=True)

frame = frame.drop(columns=["_ts"])
frame.to_csv(DST, index=False)
log.update({"output_rows": int(len(frame)), "output": str(DST)})
MANIFEST.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
print(json.dumps(log, indent=2, default=str), flush=True)
print("REPAIR_DONE", flush=True)
