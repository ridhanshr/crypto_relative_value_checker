from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import pandas as pd


URL = "https://data.binance.vision/data/spot/daily/klines/{symbol}/1d/{symbol}-1d-{date}.zip"
COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def download_binance_daily(symbols, start, end, output):
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for symbol in symbols:
        for date in dates:
            stamp = date.strftime("%Y-%m-%d")
            url = URL.format(symbol=symbol.upper(), date=stamp)
            try:
                with urlopen(url, timeout=30) as response:
                    archive = ZipFile(BytesIO(response.read()))
                    raw = archive.read(archive.namelist()[0])
            except Exception:
                continue
            frame = pd.read_csv(BytesIO(raw), header=None, names=COLUMNS)
            unit = "us" if frame.open_time.astype("int64").abs().max() > 10**14 else "ms"
            rows.append(pd.DataFrame({"timestamp": pd.to_datetime(frame.open_time, unit=unit, utc=True), "asset": symbol.upper(), "price": frame.close.astype(float)}))
    if not rows:
        raise RuntimeError("No Binance Vision files downloaded")
    result = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "asset"])
    result["asset_return"] = result.groupby("asset")["price"].pct_change()
    result["signal"] = result.groupby("timestamp")["asset_return"].shift(1)
    result = result.dropna(subset=["signal"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
