from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crypto_checker.binance_vision import download_binance_daily

download_binance_daily(
    [
        # --- 15 coin awal (top-cap, sudah diriset) ---
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
        "DOTUSDT", "TRXUSDT", "ATOMUSDT", "NEARUSDT", "UNIUSDT",

        # --- L1/L2 mid-cap ---
        "OPUSDT", "ARBUSDT", "APTUSDT", "SUIUSDT", "SEIUSDT",
        "INJUSDT", "TIAUSDT", "ALGOUSDT",

        # --- DeFi ---
        "AAVEUSDT", "MKRUSDT", "CRVUSDT", "COMPUSDT", "SNXUSDT", "DYDXUSDT",

        # --- Oracle/Infra ---
        "FILUSDT", "GRTUSDT", "RUNEUSDT", "ICPUSDT", "RENDERUSDT",

        # --- Gaming/Metaverse ---
        "SANDUSDT", "MANAUSDT", "AXSUSDT", "GALAUSDT", "IMXUSDT",

        # --- Meme (volatilitas tinggi, kurang efisien) ---
        "PEPEUSDT", "WIFUSDT", "FLOKIUSDT", "BONKUSDT",

        # --- Legacy mid-cap ---
        "VETUSDT", "HBARUSDT", "XLMUSDT", "ETCUSDT", "THETAUSDT",
    ],
    "2024-01-01",
    "2026-09-01",
    "data/binance_real_case_extended.csv",
    futures=True,
)
print("Wrote data/binance_real_case.csv")
