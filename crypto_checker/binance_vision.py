from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from zipfile import ZipFile
import pandas as pd
from .assets import canonical_asset


SPOT_URL = "https://data.binance.vision/data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
FUTURES_URL = "https://data.binance.vision/data/futures/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
FUNDING_MONTHLY_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{date}.zip"
FUNDING_API_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
VALID_INTERVALS = ("1d", "4h", "1h")


def download_binance_daily(symbols, start, end, output, futures=True, interval="1d"):
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Unsupported interval {interval}; valid: {VALID_INTERVALS}")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    dates = pd.date_range(start_ts, end_ts, freq="D")
    rows = []
    failures = []
    def fetch(item):
        symbol, date = item
        stamp = date.strftime("%Y-%m-%d")
        template = FUTURES_URL if futures else SPOT_URL
        url = template.format(symbol=symbol.upper(), date=stamp, interval=interval)
        print(f"DOWNLOAD {url}", flush=True)
        try:
            with urlopen(url, timeout=30) as response:
                archive = ZipFile(BytesIO(response.read()))
                raw = archive.read(archive.namelist()[0])
            frame = pd.read_csv(BytesIO(raw), header=None, names=COLUMNS)
            frame = frame[pd.to_numeric(frame.open_time, errors="coerce").notna()].copy()
            if frame.empty:
                return None
            frame["open_time"] = pd.to_numeric(frame["open_time"])
            unit = "us" if frame.open_time.astype("int64").abs().max() > 10**14 else "ms"
            return pd.DataFrame({
                "timestamp": pd.to_datetime(frame.open_time, unit=unit, utc=True),
                "asset": canonical_asset(symbol),
                "open": frame.open.astype(float),
                "high": frame.high.astype(float),
                "low": frame.low.astype(float),
                "price": frame.close.astype(float),
                "volume": frame.volume.astype(float),
                "quote_volume": frame.quote_volume.astype(float),
            })
        except Exception:
            failures.append(url)
            print(f"SKIP {url}", flush=True)
            return None
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures_list = [pool.submit(fetch, (symbol, date)) for symbol in symbols for date in dates]
        for future in as_completed(futures_list):
            value = future.result()
            if value is not None:
                rows.append(value)
    if not rows:
        raise RuntimeError("No Binance Vision files downloaded")
    result = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "asset"])
    result["asset_return"] = result.groupby("asset")["price"].pct_change()
    result["signal"] = result["asset_return"]
    if futures:
        funding = []

        def fetch_funding_archive(url):
            with urlopen(url, timeout=30) as response:
                archive = ZipFile(BytesIO(response.read()))
                raw = archive.read(archive.namelist()[0])
            funding_frame = pd.read_csv(BytesIO(raw))
            funding_frame.columns = [str(c).strip() for c in funding_frame.columns]
            normalized = {c.lower().replace("_", ""): c for c in funding_frame.columns}
            time_col = next((normalized[k] for k in ("calctime", "fundingtime", "fundingtimestamp") if k in normalized), None)
            rate_col = next((normalized[k] for k in ("lastfundingrate", "fundingrate", "funding") if k in normalized), None)
            if time_col is None or rate_col is None:
                raise ValueError(f"Unrecognized funding columns: {list(funding_frame.columns)}")
            return pd.DataFrame({
                "timestamp": pd.to_datetime(funding_frame[time_col], unit="ms", utc=True),
                "asset": None,
                "funding_rate": funding_frame[rate_col].astype(float),
            })

        def fetch_funding_api(symbol, range_start, range_end):
            rows = []
            cursor = int(range_start.timestamp() * 1000)
            end_ms = int(range_end.timestamp() * 1000)
            while cursor < end_ms:
                params = urlencode({"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000})
                with urlopen(FUNDING_API_URL + "?" + params, timeout=30) as response:
                    batch = json.loads(response.read())
                if not batch:
                    break
                for row in batch:
                    rows.append({"timestamp": pd.to_datetime(row["fundingTime"], unit="ms", utc=True), "funding_rate": float(row["fundingRate"])})
                latest = max(int(r["fundingTime"]) for r in batch)
                if latest < cursor:
                    raise RuntimeError(f"Funding pagination stalled for {symbol}")
                cursor = latest + 1
            return pd.DataFrame(rows)

        missing_funding = []
        for symbol in symbols:
            symbol = symbol.upper()
            first_month = start_ts.replace(day=1)
            for date in pd.date_range(first_month, end_ts, freq="MS"):
                stamp = date.strftime("%Y-%m")
                url = FUNDING_MONTHLY_URL.format(symbol=symbol, date=stamp)
                print(f"DOWNLOAD {url}", flush=True)
                try:
                    month_frame = fetch_funding_archive(url)
                except Exception:
                    print(f"SKIP {url}", flush=True)
                    month_end = min(date + pd.offsets.MonthEnd(0), end_ts)
                    day_start = max(date, start_ts)
                    if day_start > month_end:
                        continue
                    api_url = f"{FUNDING_API_URL}?symbol={symbol}"
                    print(f"DOWNLOAD {api_url} (archive missing; API fallback)", flush=True)
                    try:
                        month_frame = fetch_funding_api(symbol, day_start, month_end)
                    except Exception as exc:
                        print(f"SKIP {api_url} ({exc})", flush=True)
                        missing_funding.append((symbol, stamp))
                        continue
                month_frame["asset"] = canonical_asset(symbol)
                funding.append(month_frame)
        if missing_funding:
            missing_frame = pd.DataFrame(missing_funding, columns=["asset", "month"])
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(Path(output).with_name(Path(output).stem + "_partial.csv"), index=False)
            missing_frame.to_csv(Path(output).with_name(Path(output).stem + "_missing_funding.csv"), index=False)
            raise RuntimeError(f"Funding archives missing for {missing_funding}; partial klines and missing-funding manifest saved beside output; refusing invalid final dataset")
        if funding:
            funding_frame = pd.concat(funding, ignore_index=True)
            if interval == "1d":
                funding_frame["date"] = funding_frame["timestamp"].dt.floor("D")
                if not (funding_frame["timestamp"] < funding_frame["date"] + pd.Timedelta(days=1)).all():
                    raise RuntimeError("Funding event settled after candle close; refusing lookahead merge")
                per_day = funding_frame.groupby(["date", "asset"]).size()
                freq_report = (
                    funding_frame.assign(date=funding_frame["date"])
                    .groupby("asset")["date"]
                    .apply(lambda s: s.value_counts().max())
                    .rename("max_settlements_per_day")
                )
                print("max settlements/day per asset (values >3 mean sub-8h funding interval, aggregated by sum):", flush=True)
                over = freq_report[freq_report > 3]
                if not over.empty:
                    print(over.to_string(), flush=True)
                freq_report.to_csv(Path(output).with_name(Path(output).stem + "_funding_frequency.csv"))
                result["date"] = result["timestamp"].dt.floor("D")
                funding_frame = funding_frame.groupby(["date", "asset"], as_index=False)["funding_rate"].sum()
                result = result.merge(funding_frame, on=["date", "asset"], how="left")
                result = result.drop(columns=["date"])
            else:
                funding_frame = funding_frame.rename(columns={"timestamp": "date"})
                result = result.merge(funding_frame, left_on=["timestamp", "asset"], right_on=["date", "asset"], how="left")
                result = result.drop(columns=["date"])
        if not funding:
            raise RuntimeError("Funding data unavailable from Binance Vision/API; refusing zero funding fallback")
        coverage = result.groupby("asset")["funding_rate"].apply(lambda x: int(x.notna().sum())).to_dict()
        missing_rows = result[result["funding_rate"].isna()][["timestamp", "asset"]].copy()
        print(f"FUNDING_ROWS {int(funding_frame.funding_rate.notna().sum())} COVERAGE {coverage}", flush=True)
        if not missing_rows.empty:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(Path(output).with_name(Path(output).stem + "_partial.csv"), index=False)
            missing_rows.to_csv(Path(output).with_name(Path(output).stem + "_missing_funding_rows.csv"), index=False)
            raise RuntimeError(f"Funding coverage incomplete: {len(missing_rows)} price rows have no funding; partial dataset and gap manifest saved; refusing zero funding fallback")
    if failures:
        print(f"FAILED_FILES {len(failures)}", flush=True)
    result = result.dropna(subset=["signal"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
