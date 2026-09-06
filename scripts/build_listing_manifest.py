"""Build a historical listing manifest for Binance USDT-M futures.

Enumerates every symbol prefix ever published on Binance Vision
(``futures/um/daily/klines/<SYMBOL>/1d/``) -- including dead coins like
LUNA/FTT that no current API returns -- and records first/last daily file
as listing/delisting proxies. Cross-checks the current roster via
``fapi/.../exchangeInfo`` (network permitting).

Output: data/historical_listing_manifest.csv with columns
symbol, listed_at, delisted_at, status, source, where source is
"inferred_vision_first_seen" (listing proxy) and delisted_at is None for
symbols still TRADING on the live exchange.

Listing dates are INFERRED from first published file, not official
announcements: first file can lag the true listing by days. Good enough to
measure survivorship bias explicitly (which dead assets inhabited the
 tradable universe in-window), never to claim official listing precision.
"""

import sys
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd

S3_LIST_URL = "https://s3.ap-northeast-1.amazonaws.com/data.binance.vision"
KLINES_PREFIX = "data/futures/um/daily/klines/"
EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "crypto-relative-value-checker/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def list_vision_symbols():
    """All '<SYMBOL>/1d/' prefixes under the futures daily klines tree."""
    symbols = set()
    token = None
    while True:
        params = {"list-type": "2", "prefix": KLINES_PREFIX, "delimiter": "/"}
        if token:
            params["continuation-token"] = token
        url = S3_LIST_URL + "?" + urllib.parse.urlencode(params)
        root = ET.fromstring(_get(url))
        for cp in root.findall("s3:CommonPrefixes", NS):
            prefix = cp.findtext("s3:Prefix", default="", namespaces=NS)
            rest = prefix[len(KLINES_PREFIX):]
            symbol = rest.split("/")[0]
            if symbol:
                symbols.add(symbol)
        if root.findtext("s3:IsTruncated", default="false", namespaces=NS) != "true":
            break
        token = root.findtext("s3:NextContinuationToken", default="", namespaces=NS)
        if not token:
            break
    return sorted(symbols)


def list_symbol_files(symbol, interval="1d"):
    """All daily file dates published for one symbol (paginated)."""
    prefix = f"{KLINES_PREFIX}{symbol}/{interval}/"
    dates = set()
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        url = S3_LIST_URL + "?" + urllib.parse.urlencode(params)
        root = ET.fromstring(_get(url))
        for obj in root.findall("s3:Contents", NS):
            key = obj.findtext("s3:Key", default="", namespaces=NS)
            name = key.rsplit("/", 1)[-1]
            # <SYMBOL>-1d-YYYY-MM-DD.zip (+ optional .CHECKSUM)
            try:
                stamp = name.split(f"-{interval}-", 1)[1].split(".zip")[0]
                dates.add(stamp)
            except IndexError:
                continue
        if root.findtext("s3:IsTruncated", default="false", namespaces=NS) != "true":
            break
        token = root.findtext("s3:NextContinuationToken", default="", namespaces=NS)
        if not token:
            break
    return sorted(dates)


def live_trading_symbols():
    try:
        payload = json.loads(_get(EXCHANGE_INFO_URL))
    except Exception as exc:
        print(f"exchangeInfo unreachable ({exc}); all delisted_at stay inferred", flush=True)
        return None
    return {s["symbol"] for s in payload.get("symbols", []) if s.get("status") == "TRADING" and s.get("contractType") == "PERPETUAL"}


def build_manifest(symbols=None, progress_every=25):
    symbols = symbols or list_vision_symbols()
    print(f"{len(symbols)} vision symbols", flush=True)
    live = live_trading_symbols()
    rows = []
    for i, symbol in enumerate(symbols, 1):
        try:
            dates = list_symbol_files(symbol)
        except Exception as exc:
            print(f"  skip {symbol}: {exc}", flush=True)
            continue
        if not dates:
            continue
        listed = pd.Timestamp(dates[0], tz="UTC")
        last = pd.Timestamp(dates[-1], tz="UTC")
        still_live = bool(live is not None and symbol in live)
        rows.append({
            "symbol": symbol,
            "listed_at": listed,
            "delisted_at": None if still_live else last,
            "status": "TRADING" if still_live else "NOT_TRADING",
            "source": "inferred_vision_first_seen" + (";live_exchangeinfo" if live is not None else ";exchangeinfo_unreachable"),
        })
        if i % progress_every == 0:
            print(f"  {i}/{len(symbols)}", flush=True)
    manifest = pd.DataFrame(rows, columns=["symbol", "listed_at", "delisted_at", "status", "source"])
    return manifest.sort_values("symbol").reset_index(drop=True)


def main():
    out = Path("data/historical_listing_manifest.csv")
    manifest = build_manifest()
    manifest.to_csv(out, index=False)
    dead = manifest[manifest["status"] == "NOT_TRADING"]
    print(f"wrote {out}: {len(manifest)} symbols, {len(dead)} not trading", flush=True)
    print("sample dead:", ", ".join(dead["symbol"].head(20).tolist()), flush=True)


if __name__ == "__main__":
    main()
