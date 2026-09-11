# Crypto Relative-Value Checker

Sistem untuk menguji strategi trading crypto long-short secara hati-hati.

Sistem membandingkan banyak coin pada waktu yang sama. Coin dengan sinyal paling kuat dibeli (**long**). Coin dengan sinyal paling lemah dijual (**short**). Karena posisi long dan short dibuat bersamaan, tujuan strategi adalah mencari perbedaan performa antar-coin, bukan menebak arah seluruh market.

Sistem menghitung:

- PnL dari perubahan harga.
- PnL funding rate futures.
- Fee transaksi (flat atau bertingkat per likuiditas).
- Slippage.
- Turnover dan capacity check.
- Exposure long, short, gross, dan net.
- Sharpe ratio (dengan koreksi multiple-testing).
- Volatilitas dan vol-targeting overlay.
- Maximum drawdown dari equity awal.
- Risk violation (daily loss, max drawdown, capacity, forced exit).
- Performa per rezim market (bull/bear/sideways x high/low vol).
- Benchmark: equal-weight, BTC, ETH, long-only, market-neutral reference.

Sistem **tidak menjamin profit**. `DEPLOYABLE=True` hanya muncul jika 5 hard gate lolos (`wf_positive`, `oos_positive`, `no_risk_violations`, `no_capacity_violations`, `reality_check_pass`). Jangan pernah melonggarkan gate agar hasil menjadi `True`.

## Flow Visual

![Crypto Relative-Value Checker system flow](quant_pipeline_flow_horizontal_legend.png)

## Flow Sistem

```text
Download data (klines 1d/4h/1h + funding monthly/API)
    |
    v
Preflight validation (+ universe manifest, delisting suspects)
    |
    v
Canonical asset mapping (+ migration collision check, official price factors)
    |
    v
Signal audit + factor library (15 kandidat)
    |
    v
Ranking long-short (equal weight, vol-targeting, rebalance filter)
    |
    v
Simulasi execution dan PnL (fee/slippage tiered, funding, capacity limit)
    |
    v
Validation train/validation/OOS dengan CONTINUOUS equity
    |
    v
Walk-forward + regime test + reality check (Newey-West, bootstrap, Bonferroni)
    |
    v
Cost stress (multiplier saat tiered) + risk/capacity/spread gate
    |
    v
Deployment decision (fail-closed)
```

## Struktur Proyek

```text
crypto_checker/
  core.py            # mesin backtest: ranking, PnL, fee/slippage, vol-targeting, capacity  binance_vision.py  # downloader klines 1d/4h/1h + funding (arsip bulanan + fallback API)
  assets.py          # canonical mapping + migration factors + continuity audit
  lifecycle.py       # asset lifecycle engine: segmen listed/delisted, triase gap,
                     # manifest historis, metrik survivorship gap
  repair.py          # primitif repair dataset (relabel migrasi, drop stale, tolak duplikat)
  preflight.py       # validasi dataset sebelum backtest (fail-fast)
  signals.py         # pustaka 15 faktor (momentum, reversal, carry, low-vol, combo z-score)
  signal_audit.py    # cek signal konstan / forward-fill / kumulatif
  research.py        # rank-IC, t-stat, Newey-West, bootstrap CI, quantile return
  reality_check.py   # koreksi multiple-testing (Bonferroni) + Deflated Sharpe Ratio
  validation.py      # split train/val/OOS continuous-equity, regime, benchmark, stress
  selection.py       # walk-forward selection bebas leakage + ensemble top-k
  capacity.py        # capacity curve per level AUM (10k/100k/1M/10M)
  decision.py        # keputusan deployment gabungan semua gate
  api.py             # entry point resmi validate_csv() + envelope 3-keadaan
  gate.py            # gate otomatis v2 (APPROVED_CANDIDATE/REJECTED/FLAG_REVIEW + review fields)
  trial_registry.py  # log semua attempt + effective-N clustering
  dsr.py             # DSR v2 (fail-loud, Pearson kurtosis, gray-zone siap gate)
  cpcv.py            # combinatorial purged CV + regime labeling + summarize incl. max DD
  ensemble_check.py  # pre-check korelasi member sebelum ensemble dirakit
  repair.py          # primitif repair dataset (relabel migrasi, drop stale, tolak duplikat)
  spread_check.py    # ukur spread bid-ask riil dari order book Binance
  cli.py             # command-line interface
scripts/
  download_midcap.py # unduh dataset per sektor (l1_l2, defi, oracle_infra, gaming, meme, legacy, mega)
  download_binance_sample.py
  fetch_funding.py   # unduh + agregasi funding rate (arsip bulanan + API)
  repair_midcap.py   # repair generik dataset partial
  repair_71.py       # repair file 71-aset (contoh repair terdokumentasi + manifest)
  analyze_midcap.py  # pipeline analisis midcap end-to-end (preflight, spread, signal, WF)
  build_listing_manifest.py # bangun manifest historis 1028 simbol futures dari Binance Vision
  sweep_exposure.py  # sweep skala exposure/gross
tests/
  test_core.py       # 75 regression test
```

## Cara Penggunaan

Prinsip: **data kotor tidak pernah menghasilkan angka.** Setiap workflow di bawah berhenti di gerbang pertama yang gagal, dengan alasan eksplisit (bukan traceback misterius).

### 0. Instalasi + verifikasi sistem sehat

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Harus 115 passed. Kalau ada yang gagal, jangan lanjut — perbaiki environment dulu.

**Required pre-commit step** (belum ada CI server — harness di bawah ini manual dan wajib dijalankan sebelum push bila menyentuh kode numerik/engine):

```bash
python -m pytest tests -q                                  # L1 + golden + kontrak
python scripts/determinism_check.py                        # L2: 2 run identik
python scripts/runtime_burnin.py --stage small             # L5 kecil: 100/100
```

`runtime_burnin --stage full` (3x decision penuh) hanya untuk perubahan besar pada inti numerik. Golden fixture (`tests/fixtures/`) gagal = drift disengaja atau bug — review diff sebelum regenerate.

### 1. Jalan penuh riset + vonis deployment (ENTRY POINT RESMI)

Satu-satunya pintu resmi untuk konsumen (termasuk Quantara): `validate_csv()` / `python -m crypto_checker.validate`. Jangan panggil `core.py`, `selection.py`, atau fungsi internal satu per satu — itu detail implementasi yang boleh berubah tanpa pemberitahuan.

```bash
python -m crypto_checker.validate ^
  --input data\midcap_2y_daily_2_funded.csv ^
  --output reports\hasil_saya
```

atau dari Python:

```python
from crypto_checker.api import validate_csv
envelope = validate_csv("data/midcap_2y_daily_2_funded.csv", output_dir="reports/hasil_saya")
```

Output utama di folder output: `decision.json` (**selalu ditulis**, atomik, di semua keadaan — inilah satu-satunya file yang perlu dibaca), plus `preflight.json`, `walk_forward.json`, `validation.json`, `capacity_curve.json`, `deployment_decision.json` (detail). Baca `status` + `decision` + `deployable` + `gates` yang gagal.

| status | decision | exit | arti |
|---|---|---|---|
| `SUCCESS` | `APPROVED`/`REJECTED` | 0 | evaluasi jalan penuh; REJECTED = ditolak karena merit (gate/DSR), **bukan** error |
| `FAILED_VALIDATION` | null | 2 | data tak lolos gerbang; `metrics`/`capacity` null, `errors` terisi, `deployable` false |
| `CHECKER_ERROR` | null | 1 | crash/bug internal; tanpa klaim riset apa pun |

Exit code policy: `SUCCESS` memakai exit code 0 untuk `APPROVED` maupun `REJECTED`, dan juga untuk `FLAG_REVIEW`, karena evaluasi checker selesai. `REJECTED` adalah keputusan research, bukan kegagalan teknis. Quantara wajib membaca `status`, `decision`, `review_required`, dan `gate_decision`; jangan menyimpulkan dari exit code saja. `FAILED_VALIDATION` memakai 2; `CHECKER_ERROR` memakai 1.

### 2. Uji satu signal manual

```bash
python -m crypto_checker.cli ^
  --input data\midcap_2y_daily_2_funded.csv ^
  --output reports\lowvol14 ^
  --signal low_vol_14 --n-long 3 --n-short 3 ^
  --liquidity-column quote_volume --cost-preset midcap
```

Aturan eksekusi yang dipaksakan: **signal(t) dieksekusi di t+1** untuk semua signal tanpa kecuali. Input butuh ≥2 bar per aset (bar pertama dikorbankan untuk lag).

### 3. Dataset kotor → repair → verifikasi (skenario ideal)

Kalau preflight menolak file (duplikat / gap tak terjelaskan / diskontinuitas migrasi):

```bash
# a. Lihat apa yang salah (tidak perlu flag toleransi dulu)
python scripts\strict_gap_check.py

# b. Repair memakai primitif library (contoh terdokumentasi: repair_71.py)
python scripts\repair_71.py
# -> data\midcap_2y_daily_2_repaired.csv + *_repair_manifest.json
#    (setiap baris yang dibuang/di-relabel tercatat di manifest)

# c. Verifikasi ulang sampai valid
python scripts\verify_dataset.py --input data\midcap_2y_daily_2_repaired.csv
```

Target: `strict valid: True`, `errors: []`. Lubang 1-hari otomatis jadi warning (hiccup API); halt migrasi terdokumentasi tidak pernah error; hanya lubang multi-hari tak terjelaskan yang menggagalkan run.

### 4. Capacity: masih masuk akal di AUM berapa? (via Python)

```python
from crypto_checker.capacity import capacity_curve
from crypto_checker.core import CheckerConfig
report = capacity_curve(data, base_config=CheckerConfig(
    n_long=3, n_short=3, signal_column="low_vol_14",
    liquidity_column="quote_volume",
    liquidity_tiers=((0.5, 0.0004, 0.0005), (0.0, 0.0015, 0.004))),
    output_dir="reports/capacity")
# per level AUM: sensible True/False + headroom_multiple
# headroom < 1 = sudah breach di AUM itu
```

### 5. Survivorship: seberapa besar bias universe saya? (via Python)

```bash
python scripts\build_listing_manifest.py   # sekali saja (butuh internet)
```

```python
from crypto_checker.lifecycle import measure_survivorship_gap
gap = measure_survivorship_gap(data, "data/historical_listing_manifest.csv")
# gap["missing_dead"] = koin mati in-window yang tak ada di dataset
# (crash mereka tidak masuk cross-section -> return terflatter)
```

### Tabel kegagalan umum

| Pesan | Artinya | Tindakan |
|---|---|---|
| `PREFLIGHT_FAILED` | data tidak lolos gerbang (lihat `errors` di `preflight.json`) | workflow 3 |
| `REPAIR_REFUSED: N residual duplicates` | duplikat tersisa pasca-repair | investigasi manual, jangan agregasi diam-diam |
| `Migration source collision` | dua ticker sumber hidup bersamaan | potong di effective date resmi |
| `Migration price discontinuities require official factors` | seam >20% tanpa faktor resmi | exclude aset + dokumentasikan, atau cari faktor resmi |
| `No executable signals` | <2 bar per aset | tambah data |
| `Missing price for held positions` | aset hilang saat dipegang (mode `error`) | pakai `delist_mode="forced_exit"` untuk riset, investigasi untuk produksi |

## Download Data

Dataset utama (daily 2 tahun, semua sektor + mega-cap sebagai reference):

```bash
python scripts\download_midcap.py ^
  --start 2024-01-01 ^
  --end 2026-08-31 ^
  --interval 1d ^
  --output data\midcap_2y_daily.csv ^
  --sectors l1_l2,defi,oracle_infra,gaming,meme,legacy ^
  --include-mega
```

Interval yang didukung: `1d`, `4h`, `1h`. Untuk `1h`/`4h`, funding di-merge per candle tepat (event funding 8-jam-an menempel pada candle-nya); untuk `1d`, funding harian dijumlah dari 3 settlement. Guard lookahead menolak event yang settle setelah candle close.

Semua annualisasi (Sharpe, volatilitas, annualized return, vol-targeting) memakai `periods_per_year` yang diinferensi dari spasi timestamp — daily→365.0 persis, 4h→2190.0, 1h→8760.0 (bisa dioverride via `CheckerConfig(periods_per_year=...)`, tercatat di tiap ringkasan sebagai `periods_per_year`). Window walk-forward (`min_train_days`/`test_days`) otomatis diskala ke jumlah periode agar artinya tetap "hari" di semua interval.

Aturan funding downloader (fail-closed):

- Bulan lengkap: arsip bulanan Binance Vision.
- Bulan berjalan (arsip belum terbit): fallback API `fapi.binance.com` dengan pagination.
- Keduanya gagal: error eksplisit berisi daftar `(simbol, bulan)` yang hilang + saran (set `--end` ke bulan lengkap terakhir), dan file partial + manifest disimpan di samping output. **Tidak ada funding yang diam-diam diisi nol.**

Downloader mencatat setiap URL (`DOWNLOAD`/`SKIP`) dan simbol yang hilang/rename (mis. MATICUSDT -> POLUSDT) terlihat jelas di log.

### Troubleshooting koneksi Binance

Endpoint `data.binance.vision` dan `fapi.binance.com` kadang timeout / terblokir DNS bawaan ISP. Solusi yang terbukti: ganti DNS ke Cloudflare.

- Preferred DNS: `1.1.1.1`
- Alternate DNS: `1.0.0.1`

Cara (Windows): Settings > Network > adapter aktif > Properties > IPv4 > Use the following DNS server addresses > isi `1.1.1.1` dan `1.0.0.1` > OK, lalu ulangi download. Jika masih gagal, script mencetak `SPREAD_CHECK_UNAVAILABLE` / `SKIP` per file — jangan lanjutkan analisis seolah data lengkap.

## Preflight Validation

Sebelum backtest, `preflight.json` memeriksa: kolom wajib, timestamp UTC, harga positif-finite, signal numerik-finite, funding wajib lengkap (mode futures), `quote_volume` (mode midcap), duplikat, jumlah aset/periode minimum, gap timestamp, diskontinuitas migrasi, dan tabrakan migrasi. Output tambahan:

- `universe_membership`: kehadiran tiap aset per timestamp.
- `asset_status`: first/last seen + status (`active_full_period` / `partial_history_or_delisted`).
- `delisting_suspects`: aset yang berhenti jauh sebelum tanggal akhir — hari terakhirnya wajib diperlakukan sebagai forced exit, bukan dibuang diam-diam.
- `survivorship_note`: universe berasal dari simbol aktif saat ini; tanpa verifikasi membership independen, hasil berpotensi bias survivorship yang belum terukur.
- `lifecycle_manifest`: segmen `listed_at`/`delisted_at` per (canonical, source symbol) — otoritas universe point-in-time (lihat Asset Lifecycle Engine). Tanggal listing yang masih `inferred_from_data` diberi warning eksplisit: ganti dengan tanggal listing resmi sebelum klaim survivorship-free.

Mode eksplorasi `price_only` mentoleransi gap kecil (jadi warning + forced exit di backtest). Gap timestamp ditriase tiga jenis (`classify_timestamp_gaps`): `migration_halt` — blok kosong berisi/berdampingan effective date resmi (fakta halt bursa, selalu warning, tidak pernah error); `listing/delisting edge` — tepi series (ditangani segmen lifecycle); `unexplained_interior` — lubang 1-hari ditoleransi sebagai warning (hiccup API, forced-exit berlaku; ambang via `tolerated_gap_days`, default 1, 0 = ketat penuh); hanya lubang multi-hari yang bisa menggagalkan run futures (`--allow-gaps` hanya mentoleransi jenis ini, mode eksplorasi saja). Contoh nyata: 43 gap dataset funded (GUSDT:34, POLUSDT:9) semuanya halt migrasi kontinu (GAL halt 12 Jul 2024 → G jalan 15 Agu; MATIC halt 5 Sep → POL 13 Sep) — mode ketat lolos tanpa flag.

## Asset Lifecycle Engine (`crypto_checker/lifecycle.py`)

Fondasi universe construction: tiap aset canonical dipecah menjadi segmen `(symbol, listed_at, delisted_at, event, source)`.

- Batas migrasi (GAL→G, MATIC→POL, ...) memakai effective date resmi pengumuman (`source: official`); tanggal listing/delisting sisanya diinferensi dari first/last seen (`source: inferred_from_data` — placeholder, bukan fakta).
- `active_assets(lifecycle, ts)` menjawab "apa yang tradable di instant t" — backtest konsisten dengan ini by construction (ranking hanya memakai aset yang hadir di bar t; tidak ada forward fill).
- `audit_universe_compliance(data, lifecycle)` memeriksa dataset terhadap manifest **independen** (mis. yang dikurasi manual dari tanggal resmi) dan menandai `trading_before_listing` / `trading_after_delisting` / `no_lifecycle_segment`.
- `write/load_lifecycle_manifest` (CSV) untuk kurasi manual: ekspor manifest inferensi sebagai titik awal, koreksi dengan tanggal resmi, lalu audit ulang.
- Historical listing manifest (`scripts/build_listing_manifest.py`): enumerasi SEMUA prefix simbol futures USDT-M yang pernah ada di Binance Vision (1028 simbol, termasuk yang mati seperti LUNA/FTT/SRM yang tak lagi muncul di API) + status live via exchangeInfo → `data/historical_listing_manifest.csv` (lokal, di-ignore git). `lifecycle_from_listing_manifest` mengadaptasinya ke segmen lifecycle; `measure_survivorship_gap` mengukur bias EKSLISIT (bukan note): dataset funded 47-aset vs 934 simbol in-window → 365 dead-in-window tak tercakup (upper bound, termasuk tracker saham/ETF; contoh material: FTT, SRM, ALPHA, BNX, REEF, WAVES, LOOM). LUNA benar-benar di-exclude (mati 2022, pre-window — windowing terbukti bekerja); GAL/MATIC benar-benar di-exclude (suksesor G/POL ada di dataset — kanonikalisasi terbukti bekerja). Preflight menerima `listing_manifest=` dan menulis `survivorship_gap` + warning `SURVIVORSHIP_GAP`.

## Canonical Asset Mapping

Token yang pernah ganti nama disatukan agar history tidak pecah. Kolom `source_asset` menyimpan simbol asli untuk audit.

| Legacy | Canonical | Ratio resmi | Effective date (UTC) |
|---|---|---|---|
| GAL | G | 1 GAL = 60 G | 2024-07-19 08:00 |
| OMNI | NOM | 1 OMNI = 75 NOM | 2025-10-01 08:00 |
| MATIC | POL | 1 MATIC = 1 POL | 2024-09-13 10:00 |
| NANO | XNO | 1 NANO = 1 XNO | 2022-01-28 04:00 |
| VEN | VET | 1 VEN = 100 VET | 2018-07-25 04:00 |
| BCC | BCH | (faktor resmi belum terverifikasi) | - |
| ANTOLD | ANT | (faktor resmi belum terverifikasi) | - |

Aturan keras: overlap dua source symbol pada timestamp yang sama (`MIGRATION_COLLISION`) menghentikan run; faktor harga hanya diterapkan sebelum effective date (`adjusted = old_price / ratio`); rasio tanpa sumber resmi tidak ditebak — tetap `None` dan memicu error bila diskontinuitas terdeteksi.

## Signal (15 kandidat, semua point-in-time)

Momentum (skip 1 hari terakhir): `momentum_7/14/30`, `vol_adj_momentum_14/30`. Reversal: `reversal_1`, `resid_reversal_14/30` (residu terhadap beta BTC trailing). Low-vol anomaly: `low_vol_14/30`. Carry/funding: `carry` (-funding), `funding_surprise`, `vol_adj_carry`. Kombinasi z-score cross-sectional: `carry_mom_z`, `carry_lowvol_z`.

Definisi lag tetap: signal candle `t` dipakai untuk posisi periode `t -> t+1`. Tidak ada data masa depan.

Aturan eksekusi tunggal (timestamp = candle OPEN, `price` = close) — **signal(t) dieksekusi di t+1**, dipaksakan untuk SEMUA signal tanpa kecuali:

- Kolom signal yang di-ranking di-shift satu bar per aset SEBELUM smoothing/validasi/ranking, sehingga nilai yang di-ranking di bar `t` hanya memuat info sampai close(t-1) dan diisi di harga close(t).
- Ini menutup lubang lookahead lama: signal custom/kolom default (`signal` = return kontemporer) sebelumnya terisi di close yang sama dengan bar observasinya. Faktor bawaan yang sudah self-lag membayar satu bar lag ekstra sebagai harga aturan seragam — arah yang konservatif.
- Posisi yang diputuskan di `t` diisi di close(t) (`execution_timestamp` selalu satu bar setelah `signal_timestamp`) dan mulai menghasilkan PnL pada pergerakan `t -> t+1`. Tidak ada PnL yang diakru di bar yang informasinya menghasilkan signal. Bar pertama per aset tidak punya signal executable dan dibuang (input butuh ≥2 bar per aset, ditegakkan fail-closed). Ada regression test yang mengunci perilaku ini, termasuk devil's-advocate test bahwa spike signal di `t` baru mengubah posisi di `t+1`.

## Mesin Backtest (core)

- Ranking cross-sectional long top / short bottom, equal weight, gross default 2.0 / net 0.0.
- `signal_lookback` (smoothing per aset), `min_signal_gap` (tahan posisi bila gap ranking tipis), `rebalance_every`.
- Vol-targeting: `vol_target_annual` (0 = mati); skala = min(1, target_harian / realized_30d); masa warm-up tanpa riwayat memakai `vol_warmup_scale` 0.5.
- Biaya masuk (entry) dan rebalance sama-sama dikenakan. Mode flat (`fee_rate`/`slippage_rate`) atau bertingkat per likuiditas (`liquidity_column=quote_volume`, `liquidity_tiers`, `liquidity_lookback`). `cost_multiplier` mengalikan semua biaya (dipakai cost stress).
- Capacity: order di atas `quote_volume x max_volume_participation` dicatat sebagai violation `capacity_limit` (masuk `capacity.csv` + gate). Baris dengan volume invalid dicatat `invalid_liquidity` dan asetnya tidak bisa dipegang.
- Delisting: default `delist_mode="error"` (hard fail bila harga posisi hilang); `"forced_exit"` menutup posisi di harga terakhir (PnL periode terakhir 0, exit tetap kena biaya) dan mencatatnya — dipakai analisis eksplorasi midcap.
- Funding: long membayar bila funding positif (`funding_positive_paid_by_long`, bisa dibalik untuk eksperimen); `require_funding=True` menolak input tanpa kolom funding.
- Risk limit: `daily_loss_limit`, `max_drawdown_limit` diukur dari **running peak** intra-loop (peak diupdate tiap bar, jadi drawdown puncak-ke-lembah yang sebenarnya memicu violation, bukan cuma rugi-dari-awal); `--enforce-risk-limits` menghentikan run saat breach (mode produksi).
- Drawdown metrik dihitung dari equity awal (bukan dari equity berjalan), supaya konsisten dengan tracking intra-loop.

## Validation, Walk-Forward, Reality Check

- Split 60/20/20 dengan **continuous equity**: satu pass penuh, metrik tiap segmen dihitung dari equity awal segmen (`oos return = equity_akhir / equity_awal - 1`), bukan reset modal. Field `continuous_equity: true`.
- Cost stress: mode flat memakai tier absolut (4+5bps s/d 15+40bps); mode tiered memakai multiplier (`cost_x_2.00`, `cost_x_4.00`) supaya stress benar-benar berpengaruh.
- Walk-forward: seleksi signal x n_sides x vol-target hanya dari train tiap fold; test fold beku. `state_policy` terdokumentasi (fresh deployment per fold, khusus seleksi; keputusan live memakai validasi continuous-equity).
- Reality check: tiap baris `factor_ic` memuat `ic_tstat`, `nw_tstat` (Newey-West), bootstrap CI 95%, `adjusted_pvalue` (Bonferroni); gate `best_ic_tstat > 3.0`.
- Deflated Sharpe Ratio (Bailey & de Prado 2014): Sharpe OOS jalur seleksi dikoreksi bias data-mining atas `n_trials` konfigurasi yang dicoba (`walk_forward.json -> data_mining`: `dsr`, `expected_sharpe_null`, varians Sharpe antar-trial). DSR < 0.95 berarti seleksi kemungkinan beruntung — alpha dibunuh. Informasional (belum hard gate); juga tampil di `deployment_decision.json` sebagai `data_mining` + gate info `dsr_above_95`.
- Ensemble signal (`walk_forward(..., ensemble_top_k=k)`): per fold, top-k signal berbeda berdasar train Sharpe (masing-masing di n_sides/vol-target terbaiknya) dirata-bobot-sama di test segment — setara menjalankan tiap member di modal 1/k. Dilaporkan di `walk_forward.json -> ensemble` sebagai **diagnostik saja** (membership-nya sendiri hasil seleksi, jadi bukan estimasi unbiased; tidak pernah masuk vonis deployable). Best-signal walk-forward diperlakukan sebagai *candidate alpha*, bukan final alpha.
- Capacity curve (`crypto_checker/capacity.py::capacity_curve`): backtest diulang di AUM 10k/100k/1M/10M USD; per level dilaporkan return/Sharpe/drawdown, `breach_trades`, `breached_notional`, `max_participation_ratio`, `headroom_multiple` (berapa kali lipat AUM bisa tumbuh sebelum breach pertama; <1 = sudah breach). Tanpa kolom volume → `status: no_liquidity_data` (menolak mengarang likuiditas).
- Benchmark OOS: equal-weight basket, `btc_buy_hold`, `eth_buy_hold`, long-only equal-weight (jujur, tanpa clip), market-neutral reference. (Versi lama mem-`clip` return negatif harian ke 0 — return fabrikasi yang mustahil dicapai portfolio long-only; sudah diperbaiki.)
- Regime dari BTC 30-hari: bull/bear/sideways x hi/lo vol, dengan cutoff vol memakai **expanding quantile** (hanya data sampai waktu itu — kuantil full-sample sebelumnya membocorkan info masa depan ke label masa lalu).

## Spread Check

```bash
python -c "from crypto_checker.spread_check import check_order_book_spread; check_order_book_spread(['BTCUSDT','INJUSDT'], output='data/spread.csv')"
```

Mengukur spread bid-ask riil (bps) dan menyimpan CSV. Hasil terukur (snapshot): median ~2.3 bps, maks ~16 bps (DYDX saat volatil); tier bawah midcap (5+40 bps) konservatif ~11x di atas median spread tier bawah. Catatan: snapshot kondisi tenang — stress 2x/4x menutupi pelebaran saat krisis. Mode `slippage-mode="spread"` memakai `max(full spread terukur, floor tier)` per aset (fail-closed bila file hilang); flag `--spread-csv` menunjuk ke file spread. Simbol hilang dari bookTicker (mis. delisted) dilaporkan eksplisit. Jika API tidak terjangkau, hasil `SPREAD_CHECK_UNAVAILABLE` dan analisis otomatis turun ke mode eksplorasi (tidak deployable).

## Analisis Midcap End-to-End

```bash
python scripts\analyze_midcap.py ^
  --input data\midcap_2y_daily.csv ^
  --output reports\midcap_2y_daily ^
  --require-funding ^
  --interval 1d
```

Tanpa `--require-funding` (dataset partial tanpa funding), script berjalan mode `price_only` eksplorasi dengan forced exit — valid untuk riset IC, **tidak valid** untuk keputusan futures. Script menguji `low_vol_14/30`, `reversal_1`, `resid_reversal_14/30` dengan cost midcap + walk-forward, lalu menulis `analysis_summary.json` berisi mode, manifest universe, forced-exit assets, preflight, status spread, dan keputusan deploy.

## Menjalankan Checker Manual

```bash
python -m crypto_checker.cli ^
  --input data\midcap_2y_daily.csv ^
  --output reports\midcap_low_vol_30 ^
  --signal low_vol_30 --n-long 3 --n-short 3 ^
  --liquidity-column quote_volume --cost-preset midcap
```

Opsi penting: `--signal` (kolom signal), `--signal-lookback`, `--min-signal-gap`, `--enforce-risk-limits`, `--liquidity-column`, `--cost-preset {liquid,midcap}`, `--slippage-mode {tier,spread}`, `--spread-csv` (wajib bila mode spread), `--research` (pipeline riset penuh + deployment decision), `--skip-walk-forward`. Tier midcap: fee 5 bps flat + slippage 5/10/20/40 bps per persentil volume; tiap segmen validasi menandai `turnover_flag: OVER` bila turnover rata-rata >0,15/hari.

## Membaca Hasil

- `summary.json`: return, Sharpe, volatilitas, max drawdown, turnover, fee, slippage, funding PnL, capacity violations.
- `validation.json`: performa train/val/OOS (continuous), violations, cost stress, benchmark, regime, gates, `deployable`.
- `walk_forward/walk_forward.json`: pilihan per fold, return agregat, stress, gates, `data_mining` (DSR), `ensemble` (bila diminta).
- `capacity_curve.json` (bila dijalankan): kelayakan strategi per level AUM.
- `lifecycle_manifest.csv` (via `write_lifecycle_manifest`): segmen universe untuk kurasi manual.
- `analysis_summary.json` (analyze_midcap): mode, manifest, forced exits, status spread.

## Kontrak Output (schema v1.1, untuk integrasi)

Aturan versi (beku): **v1.0 → v1.1 = tambah field (backward-compatible), `schema_version` tetap 1.** Naik ke 2 hanya bila rename/hapus field, ubah tipe, atau ubah semantik. Konsumen lama yang membaca field lama tidak pecah oleh v1.1.

Setiap artefak JSON membawa `schema_version: 1`. Aturan: **tambah field = minor (boleh)**; ubah/hapus field terkunci = major (test kontrak gagal sampai versi di-bump). Quantara wajib menjalankan `validate_output_schema(artefak, kind)` pada setiap artefak yang dibaca.

`deployable: true` berarti **"lolos validation criteria checker" — BUKAN izin live trading / real money.** Promosi ke paper/live adalah keputusan Quantara dengan kriteria tersendiri (kalimat ini juga dikunci sebagai `deployable_meaning` di setiap decision).

| Artefak | Field terkunci (tipe) |
|---|---|
| `deployment_decision.json` (`decision`) | `schema_version, best_signal_walk_forward: str, best_n_sides, gates: dict, hard_gates: list, deployable: bool, walk_forward: dict, validation_best_signal: dict, reality_check: dict` |
| `validation.json` (`validation`) | `schema_version, splits: dict, performance: dict, gates: dict, regimes: list, continuous_equity: bool` |
| `walk_forward.json` (`walk_forward`) | `schema_version, candidates: list, n_folds, folds: list, walk_forward: dict, gates: dict, deployable: bool` (+ opsional `data_mining`, `ensemble`) |
| `capacity_curve.json` (`capacity`) | `schema_version, status: str, levels: list` (per level: `aum, total_return, sharpe, max_drawdown, capacity_violations, breach_trades, max_participation_ratio, headroom_multiple, sensible: bool`) |
| `preflight.json` (`preflight`) | `schema_version, valid: bool, errors: list, warnings: list` |
| `decision.json` (`result`, KANONIK) | `schema_version, status: SUCCESS\|FAILED_VALIDATION\|CHECKER_ERROR, decision: APPROVED\|REJECTED\|null, deployable: bool, deployable_meaning: str, review_required: bool, gate_decision: dict ({status: APPROVED_CANDIDATE\|REJECTED\|FLAG_REVIEW, reasons, reviewed_by\|null, review_decision\|null, review_timestamp\|null}), gates: dict, metrics: dict\|null (wf_total_return, wf_sharpe, wf_drawdown, oos_total_return, oos_sharpe, oos_drawdown, dsr\|null, turnover_daily), capacity: dict\|null ({max_sensible\|null, status}), risk: dict, data_quality: dict, warnings: list, errors: list, artifacts: dict` |
| `analysis_summary.json` (`analysis`, via analyze_midcap) | `schema_version, mode: str, assets: list, preflight: dict, walk_forward: dict, deployable: bool` (divalidasi + ditulis atomik; violation = run gagal) |

## Arti DEPLOYABLE

`DEPLOYABLE=True` berarti lolos semua gate, **bukan** jaminan profit. Vonis hanya memakai 5 hard gate — `wf_positive`, `oos_positive`, `no_risk_violations`, `no_capacity_violations`, `reality_check_pass` (IC terbaik lolos koreksi multiple-testing) — gate lain informasional. Satu hard gate gagal -> `DEPLOYABLE False`.

## Recheck Final via Entry Point Resmi (script `recheck_final.py`, kontrak v1.1)

Kedua dataset benchmark dijalankan ulang penuh lewat `validate_csv()` — sekaligus dogfooding kontrak Quantara. Output lokal `reports/recheck_final/{47,71}/decision.json` (reports tidak masuk git).

| | 47-aset funded | 71-aset repaired |
|---|---|---|
| status / decision | `SUCCESS` / `REJECTED` | `SUCCESS` / `REJECTED` |
| WF return / Sharpe / DD | +52% / 0,85 / −37% | −2,7% / 0,16 / −49% |
| OOS return / Sharpe / DD | −10% / −0,34 / −0,36 | −26% / −1,20 / −0,31 |
| DSR / trials after fixed-registry full-search recompute | **≈0,0** (`2.43e-12`); raw 75 → valid 65 → effective 13; null 0,91 → DSR kill | **0,0**; raw 225 → valid 195 → effective 29; null 1,54 → kill total |
| turnover harian | 0,19 | 0,26 |
| capacity.max_sensible | 100.000 USD | 100.000 USD |
| data_quality | 43 halt, 0 unexplained, 365 dead | 42 halt, 2 tolerated, 364 dead |

Low_vol_14 sempat dihitung salah dengan 5 fold sebagai trial. Recompute yang benar memakai seluruh search log: `raw=75`, `valid=65`, `effective=13`, DSR `2.43e-12` (null 0,91). Jadi selection-bias correction tetap membunuh sinyal; verdict juga ditolak oleh DD (CPCV train −46%, OOS −37%). **Low_vol_14 adalah kill total dalam bentuk saat ini, bukan kandidat yang "diselamatkan" dengan vol-targeting.** Perubahan sizing adalah hipotesis baru dengan trial budget dan registry sendiri; jika dicoba, setiap varian menambah multiple-testing burden. Vol_adj_momentum_30 tetap REJECTED dari dua arah independen: DSR 0,0 + ruin tak termodel (CPCV DD −200%/−338%, `ruin_flag=true`).

Catatan metodologi: `data_mining.observed_sharpe` (0,5128 pada selection path) dan `cpcv_summary.sharpe_mean` (sekitar 0,88 pada rata-rata OOS per-path) mengukur basis berbeda. Yang pertama adalah Sharpe agregat dari jalur pencarian/walk-forward yang dipilih dan dipakai DSR; yang kedua adalah rata-rata Sharpe OOS dari 45 split CPCV purge/embargo. Keduanya tidak boleh dibandingkan sebagai duplikat metrik atau dianggap bug.

Catatan penting: DSR menguji apakah kandidat yang dipilih survive selection bias; CPCV Sharpe menguji robustness OOS per split. Vol-targeting dapat mengubah DD, tetapi tidak mengubah fakta bahwa sinyal kandidat gagal DSR.

Catatan lama (preflight menolak CSV tanpa kolom `signal`) sudah kedaluwarsa: API resmi menyuntik kolom kerja otomatis bila belum ada. Quantara/CI wajib membaca `decision.json` (`status`, `decision`, `review_required`, `gate_decision`), bukan menyimpulkan APPROVED/REJECTED dari exit code.

## Follow-up sebelum Integrasi Operational

- `data_quality.py` belum menjadi modul mandiri; `data_quality_report` masih dict opsional dari preflight. Implementasi berikutnya wajib mengekspos `dead_asset_days`, `dead_asset_count`, `halt_count`, `halt_unexplained_count`, dan `halt_tolerated_count` lalu mengaktifkan FLAG_REVIEW secara otomatis.
- GitHub Actions menjalankan `python -m pytest tests -q` pada setiap push dan pull request. Sebelum perubahan numerik tetap jalankan lokal `determinism_check.py` dan `runtime_burnin.py --stage small`; CI tidak mengakses dataset research besar.
- Checker kini menyediakan adapter state machine fail-closed di `strategy_state.py`: RESEARCH → REJECTED/FLAG_REVIEW/APPROVED_CANDIDATE → APPROVED_PAPER → APPROVED_LIVE, dengan actor, timestamp, reason, dan event history. Quantara tetap harus menyimpan state/event ini di DB dan menerapkan operational controls sendiri; checker tidak pernah memberi izin live money secara otomatis.

## Batasan

- Backtest bukan jaminan profit live; hasil OOS bisa kena regime shift.
- Universe berpotensi bias survivorship (tercatat di setiap report).
- Snapshot spread bukan kondisi stress; slippage real-time bisa lebih besar. Liquidity crisis behaviour partially approximated via spread stress multipliers (cost stress 2x/4x) — pelebaran spread saat panik belum diukur langsung.
- Migration tanpa faktor resmi ditolak — jangan menebak rasio.
- Paper trading 60-90 hari + kill switch + rekonsiliasi order tetap wajib sebelum modal nyata.
