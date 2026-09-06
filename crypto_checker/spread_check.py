from pathlib import Path
from urllib.request import urlopen
import json
import pandas as pd


BOOK_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"


def check_order_book_spread(symbols, output="data/spread_check.csv"):
    try:
        with urlopen(BOOK_TICKER_URL, timeout=30) as response:
            tickers = json.loads(response.read())
    except Exception as exc:
        print(f"SPREAD_CHECK_UNAVAILABLE {exc}", flush=True)
        print("Jangan lanjut ke IC/backtest midcap sebelum spread riil terukur; biaya asumsi akan meremehkan edge palsu.", flush=True)
        return None
    wanted = {s.upper() for s in symbols}
    rows = []
    for t in tickers:
        symbol = t.get("symbol", "").upper()
        if symbol not in wanted:
            continue
        bid, ask = float(t["bidPrice"]), float(t["askPrice"])
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = (bid + ask) / 2
        rows.append({"asset": symbol, "bid": bid, "ask": ask, "mid": mid, "spread_bps": (ask - bid) / mid * 1e4})
    if not rows:
        print(f"SPREAD_CHECK_EMPTY requested={len(wanted)}; symbols may not exist on futures", flush=True)
        return None
    result = pd.DataFrame(rows).sort_values("spread_bps")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(f"SPREAD_CHECK rows={len(result)} median_bps={result.spread_bps.median():.2f} max_bps={result.spread_bps.max():.2f}", flush=True)
    print(result.to_string(index=False), flush=True)
    return result
