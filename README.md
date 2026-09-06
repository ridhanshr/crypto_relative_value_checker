# Crypto Relative-Value Checker

Sistem ini dipakai untuk menguji strategi trading crypto long-short secara lebih hati-hati.

Sistem membandingkan banyak coin pada waktu yang sama. Coin dengan sinyal paling kuat dibeli atau **long**. Coin dengan sinyal paling lemah dijual atau **short**. Karena posisi long dan short dibuat bersamaan, tujuan strategi adalah mencari perbedaan performa antar-coin, bukan menebak arah seluruh market.

Sistem menghitung:

- PnL dari perubahan harga.
- PnL funding rate futures.
- Fee transaksi.
- Slippage.
- Turnover.
- Exposure long, short, gross, dan net.
- Sharpe ratio.
- Volatilitas.
- Maximum drawdown.
- Risk violation.
- Performa pada kondisi market berbeda.

Sistem **tidak menjamin profit**. `DEPLOYABLE=True` hanya boleh muncul jika seluruh pemeriksaan risiko dan validasi berhasil.

## Flow Visual

![Crypto Relative-Value Checker system flow](quant_pipeline_flow_horizontal_legend.png)

## Flow Sistem

Alur utama sistem:

```text
Download data
    |
    v
Preflight validation
    |
    v
Canonical asset mapping
    |
    v
Signal audit
    |
    v
Hitung signal
    |
    v
Ranking long-short
    |
    v
Simulasi execution dan PnL
    |
    v
Validasi train / validation / out-of-sample
    |
    v
Walk-forward dan regime test
    |
    v
Cost stress dan risk gate
    |
    v
Deployment decision
```

### 1. Download Data

Downloader mengambil data futures Binance Vision.

Data yang diambil:

- Harga OHLC.
- Volume.
- Quote volume.
- Funding rate.
- Signal dasar dari return harga.

Setiap URL yang sedang diproses ditampilkan di terminal. Jika file gagal diambil, sistem menampilkan `SKIP`.

### 2. Preflight Validation

Sebelum backtest dimulai, sistem mengecek:

- Kolom wajib tersedia.
- Timestamp valid.
- Harga lebih besar dari nol.
- Signal bukan `NaN` atau `inf`.
- Funding rate lengkap jika strategi memakai futures.
- Tidak ada duplicate `(timestamp, asset)`.
- Coverage tanggal tiap asset.
- Quote volume tersedia untuk cost model midcap.
- Tidak ada masalah migration token.

Hasil preflight disimpan dalam:

```text
reports/<nama_report>/preflight.json
```

Jika validasi gagal, sistem berhenti. Data yang rusak tidak diteruskan ke checker.

### 3. Canonical Asset Mapping

Beberapa token pernah berganti nama. Sistem menyatukan history lama dan baru agar tidak dianggap sebagai dua coin berbeda.

Mapping yang tersedia:

```text
GALUSDT     -> GUSDT
OMNIUSDT    -> NOMUSDT
MATICUSDT   -> POLUSDT
NANOUSDT    -> XNOUSDT
VENUSDT     -> VETUSDT
BCCUSDT     -> BCHUSDT
ANTOLDUSDT  -> ANTUSDT
```

Symbol asli tetap disimpan pada kolom `source_asset`. Nama canonical disimpan pada kolom `asset`.

Jika migration membuat lonjakan harga yang tidak wajar dan faktor resminya belum tersedia, sistem menolak dataset.

### 4. Signal

Sistem menyediakan beberapa signal:

- `momentum_7`
- `momentum_14`
- `momentum_30`
- `vol_adj_momentum_14`
- `vol_adj_momentum_30`
- `reversal_1`
- `resid_reversal_14`
- `resid_reversal_30`
- `low_vol_14`
- `low_vol_30`
- `carry`
- `funding_surprise`
- `vol_adj_carry`
- `carry_mom_z`
- `carry_lowvol_z`

Signal selalu dihitung menggunakan data yang sudah tersedia pada waktu tersebut. Sistem tidak boleh memakai data masa depan.

Definisi lag tetap:

```text
Signal pada candle t digunakan untuk posisi periode t sampai t+1.
```

### 5. Ranking dan Portfolio

Pada setiap timestamp:

- Asset dengan ranking signal tertinggi masuk long.
- Asset dengan ranking signal terendah masuk short.
- Bobot dibagi equal-weight.
- Gross exposure dan net exposure dicatat.

Contoh konfigurasi:

```text
3 asset long
3 asset short
gross exposure = 2.0
net exposure = 0.0
```

### 6. PnL dan Biaya

PnL terdiri dari:

```text
total PnL = price PnL + funding PnL - fee - slippage
```

Untuk asset midcap, sistem dapat memakai biaya berdasarkan likuiditas. Asset dengan quote volume rendah mendapat fee dan slippage lebih tinggi.

Preset biaya:

```text
midcap:
top liquidity       = 4 bps fee + 5 bps slippage
middle liquidity    = 8-12 bps fee + 10-20 bps slippage
low liquidity       = 15 bps fee + 40 bps slippage
```

Biaya entry dan rebalance sama-sama dihitung.

### 7. Validation

Data dibagi menjadi tiga bagian:

```text
train          = 60%
validation     = 20%
out-of-sample  = 20%
```

Parameter tidak boleh dipilih menggunakan data out-of-sample.

Sistem juga menjalankan:

- Cost stress test.
- Walk-forward test.
- Market regime test.
- Benchmark equal-weight.
- Risk violation check.

## Instalasi

Pastikan Python sudah terpasang. Install dependency:

```bash
python -m pip install -r requirements.txt
```

Jalankan test:

```bash
python -m pytest tests -q
```

## Download Dataset Midcap

Download dataset daily dua tahun dengan beberapa kategori coin:

```bash
python scripts\download_midcap.py ^
  --start 2024-01-01 ^
  --end 2026-08-31 ^
  --interval 1d ^
  --output data\midcap_2y_daily.csv ^
  --sectors l1_l2,defi,oracle_infra,gaming,meme,legacy ^
  --include-mega
```

`--include-mega` menambahkan coin besar seperti BTC dan ETH. Coin tersebut berguna sebagai reference asset dan benchmark.

Untuk data per jam:

```bash
python scripts\download_midcap.py ^
  --start 2024-01-01 ^
  --end 2026-08-31 ^
  --interval 1h ^
  --output data\midcap_2y_hourly.csv ^
  --sectors l1_l2,defi,oracle_infra,gaming,meme,legacy ^
  --include-mega
```

Data hourly jauh lebih besar. Jalankan setelah dataset daily selesai dan valid.

Jika funding tidak lengkap, downloader menyimpan:

```text
data/<nama>_partial.csv
```

File partial tidak boleh digunakan untuk backtest funding.

## Analisis Dataset Midcap

Setelah dataset valid tersedia:

```bash
python scripts\analyze_midcap.py ^
  --input data\midcap_2y_daily.csv ^
  --output reports\midcap_2y_daily ^
  --require-funding ^
  --interval 1d
```

Script akan:

1. Menyatukan nama token lama dan baru.
2. Membuang asset dengan coverage tanggal tidak penuh.
3. Menjalankan preflight validation.
4. Mencoba mengambil spread order book.
5. Menguji signal `low_vol_14`.
6. Menguji signal `low_vol_30`.
7. Menguji signal `reversal_1`.
8. Menguji signal `resid_reversal_14`.
9. Menguji signal `resid_reversal_30`.
10. Menjalankan walk-forward.
11. Menyimpan report tiap signal.

Output utama:

```text
reports/midcap_2y_daily/preflight.json
reports/midcap_2y_daily/<signal>/summary.json
reports/midcap_2y_daily/walk_forward/walk_forward.json
reports/midcap_2y_daily/analysis_summary.json
```

## Menjalankan Crypto Checker

Contoh dengan signal `momentum_30`:

```bash
python -m crypto_checker.cli ^
  --input data\midcap_2y_daily.csv ^
  --output reports\midcap_momentum_30 ^
  --signal momentum_30 ^
  --n-long 3 ^
  --n-short 3 ^
  --liquidity-column quote_volume ^
  --cost-preset midcap
```

Contoh dengan signal `low_vol_30`:

```bash
python -m crypto_checker.cli ^
  --input data\midcap_2y_daily.csv ^
  --output reports\midcap_low_vol_30 ^
  --signal low_vol_30 ^
  --n-long 3 ^
  --n-short 3 ^
  --liquidity-column quote_volume ^
  --cost-preset midcap
```

Mode riset lengkap:

```bash
python -m crypto_checker.cli ^
  --input data\midcap_2y_daily.csv ^
  --output reports\midcap_research ^
  --research
```

Untuk mode production yang langsung berhenti saat risk limit terlampaui:

```bash
python -m crypto_checker.cli ^
  --input data\midcap_2y_daily.csv ^
  --output reports\midcap_production_check ^
  --signal low_vol_30 ^
  --enforce-risk-limits
```

## Membaca Hasil

Lihat file:

```text
reports/<nama_report>/summary.json
```

Metrik penting:

- `total_return`: hasil akhir strategi.
- `sharpe`: return dibanding risiko.
- `max_drawdown`: penurunan terbesar dari equity tertinggi.
- `average_turnover`: seberapa sering portfolio berubah.
- `total_fees`: total fee transaksi.
- `total_slippage`: estimasi kerugian karena slippage.
- `funding_pnl`: hasil atau biaya funding futures.
- `violations`: pelanggaran risk limit.

File:

```text
reports/<nama_report>/validation.json
```

menunjukkan hasil train, validation, out-of-sample, regime, dan cost stress.

## Arti DEPLOYABLE

`DEPLOYABLE=True` bukan berarti profit pasti di masa depan. Artinya sistem melewati semua pemeriksaan yang ditetapkan.

Gerbang deployment memerlukan:

- Train positif.
- Validation positif.
- Out-of-sample positif.
- Sharpe out-of-sample memadai.
- Drawdown tidak melewati batas.
- Cost stress tetap positif.
- Tidak ada risk violation.
- Funding coverage lengkap.
- Data tidak memiliki gap kritis.
- Migration token sudah benar.
- Spread order book tersedia dan masuk akal.

Jika salah satu syarat gagal, sistem harus tetap menghasilkan:

```text
DEPLOYABLE False
```

Jangan mengubah gate hanya agar hasil menjadi `True`. Cari signal, data, atau model execution yang lebih baik.

## Batasan

- Backtest bukan jaminan profit live.
- Spread order book dapat berubah cepat.
- Slippage midcap dapat lebih besar daripada estimasi.
- Token delisting dapat membuat coverage tidak lengkap.
- Migration token membutuhkan faktor harga resmi.
- Funding rate dapat berubah ekstrem.
- Hasil OOS juga dapat terkena regime shift.
- Paper trading tetap wajib sebelum modal nyata.

Deployment live sebaiknya dimulai dari paper trading selama 60-90 hari dengan ukuran posisi kecil, monitoring aktif, kill switch, dan rekonsiliasi order.
