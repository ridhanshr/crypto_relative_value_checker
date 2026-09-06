import argparse
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crypto_checker.binance_vision import download_binance_daily
from crypto_checker.spread_check import check_order_book_spread
from crypto_checker.assets import canonicalize_assets, audit_asset_continuity

SECTORS = {
    "l1_l2": ["INJUSDT", "SEIUSDT", "TIAUSDT", "ARBUSDT", "OPUSDT", "MATICUSDT", "POLUSDT", "SUIUSDT", "APTUSDT", "STXUSDT", "CFXUSDT"],
    "defi": ["UNIUSDT", "AAVEUSDT", "MKRUSDT", "LDOUSDT", "CRVUSDT", "COMPUSDT", "SNXUSDT", "SUSHIUSDT", "DYDXUSDT", "GMXUSDT", "PENDLEUSDT", "JTOUSDT"],
    "oracle_infra": ["LINKUSDT", "PYTHUSDT", "API3USDT", "BANDUSDT", "GRTUSDT", "ARUSDT", "FILUSDT", "STORJUSDT", "THETAUSDT", "OMNIUSDT", "NOMUSDT"],
    "gaming": ["GALUSDT", "GUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT", "IMXUSDT", "ENJUSDT", "APEUSDT", "FLOWUSDT"],
    "meme": ["DOGEUSDT", "1000SHIBUSDT", "1000PEPEUSDT", "1000BONKUSDT", "WIFUSDT", "FLOKIUSDT", "ORDIUSDT", "1000SATSUSDT"],
    "legacy": ["LTCUSDT", "BCHUSDT", "ETCUSDT", "XLMUSDT", "ALGOUSDT", "VETUSDT", "EOSUSDT", "XTZUSDT", "IOTAUSDT", "NEOUSDT", "DASHUSDT", "ZECUSDT"],
    "mega": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "TRXUSDT", "ATOMUSDT", "NEARUSDT", "LTCUSDT", "UNIUSDT"],
}

parser = argparse.ArgumentParser(description="Download midcap futures dataset from Binance Vision")
parser.add_argument("--start", required=True)
parser.add_argument("--end", required=True)
parser.add_argument("--output", default="data/midcap.csv")
parser.add_argument("--interval", default="1d", choices=["1d", "4h", "1h"])
parser.add_argument("--sectors", default="l1_l2,defi,oracle_infra,gaming,meme,legacy")
parser.add_argument("--include-mega", action="store_true", help="Also include mega-cap reference assets (needed for beta/quality filter)")
parser.add_argument("--skip-spread-check", action="store_true")
args = parser.parse_args()

symbols = []
for sector in args.sectors.split(","):
    symbols.extend(SECTORS.get(sector.strip(), []))
if args.include_mega:
    symbols.extend(SECTORS["mega"])
symbols = sorted(set(symbols))
print(f"SYMBOLS {len(symbols)} interval={args.interval} start={args.start} end={args.end}", flush=True)

if not args.skip_spread_check:
    check_order_book_spread(symbols, output=Path(args.output).with_name("spread_check.csv"))

download_binance_daily(symbols, args.start, args.end, args.output, futures=True, interval=args.interval)
downloaded = pd.read_csv(args.output)
canonical = canonicalize_assets(downloaded)
gaps = audit_asset_continuity(canonical, expected_frequency="D" if args.interval == "1d" else args.interval.upper())
if not gaps.empty:
    gaps.to_csv(Path(args.output).with_name(Path(args.output).stem + "_asset_gaps.csv"), index=False)
    raise RuntimeError(f"Asset continuity gaps detected: {len(gaps)}; gap manifest saved")
canonical.to_csv(args.output, index=False)
print(f"Wrote {args.output}", flush=True)
