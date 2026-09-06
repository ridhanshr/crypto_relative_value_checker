# Crypto Relative-Value Checker

Independent checker untuk audit strategi cross-sectional long-short.

## Jalankan

```bash
python -m crypto_checker.cli --input data/sample_signals.csv --output reports
python -m pytest
python scripts/download_binance_sample.py
python -m crypto_checker.cli --input data/binance_real_case.csv --output reports/binance_real_case
```

Input CSV wajib: `timestamp,asset,price,signal`. `target_weight` opsional; checker membangun posisi equal-weight dari ranking.

Konvensi MVP: signal pada candle `t` dieksekusi pada candle `t+1`, linear USDT, price return, fee dan slippage atas notional transaksi, funding positif dibayar posisi long.

`scripts/download_binance_sample.py` mengambil daily spot klines publik Binance Vision untuk periode 2026-08-01 sampai 2026-09-01. Signal contoh berasal dari return candle sebelumnya, bukan prediksi masa depan. URL dataset: https://data.binance.vision/. Dokumentasi OKX dapat dipakai nanti untuk adapter exchange kedua: https://www.okx.com/docs-v5/en.
