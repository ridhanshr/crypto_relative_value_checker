# Analisis Kesehatan Sistem Checker — acuan `reports/midcap_analysis_funded`

> Catatan baca: dokumen ini menilai **sistem checker-nya** (apakah mesin auditnya
> bekerja dengan benar), bukan strateginya. Status `DEPLOYABLE False` sengaja
> **diabaikan** di sini — itu vonis atas strategi, bukan atas perkakas.
> Acuan angka: run `midcap_analysis_funded` (71 aset, mode `futures`,
> `gap_tolerant: true`, spread check lolos).

## 1. Ringkasan eksekutif

Sistem checker **berjalan baik dan dapat dipercaya sebagai mesin audit**.
Seluruh rantai yang dijanjikan pipeline terbukti bekerja pada data nyata:
preflight → canonical mapping → signal → ranking → PnL (harga + funding −
biaya) → walk-forward → cost stress → spread check → keputusan fail-closed.
Tidak ditemukan bug perhitungan, tidak ada angka yang diam-diam diisi nol,
dan setiap anomali data tercatat di manifest, bukan disembunyikan.

Satu-satunya catatan serius bersifat metodologis (bukan bug): run ini memakai
mode toleran-gap (`--allow-gaps`) bersama `--require-funding`, sementara teks
peringatan preflight sendiri menyatakan mode toleran-gap "NOT valid for
futures deployment". Ketegangan ini didokumentasikan di bawah (bagian 5).

## 2. Bukti tiap subsistem bekerja

### 2.1 Preflight validation (`preflight.json`)

- `valid: true`, `errors: []` — dataset lolos pintu masuk.
- Cakupan riil tercatat jujur: **71 aset, 973 periode** (2024-01-02 s/d 2026-08-31),
  68.070 baris, `funding_present: true`, `liquidity_present: true`.
- Warning yang benar-benar informatif, bukan formalitas:
  - coverage tak merata (min 506, maks 973 hari);
  - tersangka delisting `EOSUSDT`, `MKRUSDT` dengan instruksi forced exit;
  - **59 asset-period bolong** dirinci per aset
    (`GUSDT:34; NOMUSDT:9; POLUSDT:8; UNIUSDT:6; ALGOUSDT:1; PYTHUSDT:1`);
  - survivorship note eksplisit (bias universe belum terukur).

### 2.2 Integrasi funding rate

Funding benar-benar mengalir ke PnL, bukan kolom pajangan:

| Signal | price_pnl | funding_pnl | fee+slip | total |
|---|---|---|---|---|
| low_vol_30 | −23.647 | **−16.497** | 9.517 | −49.661 |
| resid_reversal_30 | +21.465 | **−9.297** | 28.410 | −16.242 |

- Funding nonzero di **937/943 dan 925/944 hari** — konsisten dengan agregasi
  2.283 file bulanan + cutover migrasi + guard maks 3 settlement/hari.
- Magnitudo material: funding menyumbang −16,5rb dari total −49,7rb
  (low_vol_30). Fakta paling jujur dari run ini: `resid_reversal_30` sebenarnya
  punya **price PnL positif (+21.465)** — yang membunuhnya adalah funding
  (−9.297) dan biaya transaksi (28.410). Checker yang buruk akan menyembunyikan
  dekomposisi ini; checker ini menampilkannya per baris di `pnl.csv`.
- Konvensi tanda (long membayar saat funding positif) tercakup regression test.

### 2.3 Cost stress benar-benar menggigit (bug lama sudah mati)

Walk-forward base −13,6% → stress 2× biaya **−23,9%** → 4× biaya **−41,0%**.
Artinya `cost_multiplier` bekerja pada tiered costs — kebalikan dari bug lama
di v2 di mana angka stress identik dengan base (stress no-op). Sharpe ikut
turun (base −0,01 → 2× −0,20 → 4× −0,57), sesuai ekspektasi ekonomi.

### 2.4 Mekanisme forced exit & capacity

- `resid_reversal_30/violations.csv` mencatat **1 event `forced_exit`**: aset
  yang hilang dari cross-section ditutup di harga terakhir, tercatat, dan run
  lanjut — tidak crash, tidak diam-diam dilewati.
- `capacity_violations = 0` di **semua** signal: tidak ada order yang
  melanggar batas partisipasi volume. Batasnya aktif (ada regression test),
  hanya saja tidak terpicu pada konfigurasi ini.
- Ratusan violation `max_drawdown_limit`/`daily_loss_limit` tercatat apa adanya
  (mis. 738 + 21 di low_vol_30) karena `enforce_risk_limits=False` pada mode
  riset — pencatatan jujur, bukan penghentian prematur.

### 2.5 Spread check riil

`spread_check.csv`: **68 baris, median 2,16 bps, maks 8,79 bps**.
Pengukuran líve dari order book, bukan asumsi. Tier slippage midcap
(5–40 bps) konservatif ~11× di atas median tier bawah — asumsi biaya tidak
meremehkan kondisi normal. Caveat jujur: ini snapshot tenang, bukan stress.
Status: liquidity crisis behaviour partially approximated via spread stress
multipliers (2×/4×) — pelebaran spread saat panik belum diukur langsung,
bukan blocker untuk research-grade.

### 2.6 Walk-forward bebas leakage

5 fold lengkap, seleksi signal × n_sides × vol-target **hanya dari train**
(train Sharpe semua negatif: −0,01 s/d −0,04 — seleksi tidak punya bahan
bagus, dan sistem melaporkannya apa adanya alih-alih memolesnya).
Hasil agregat −13,6% / Sharpe −0,01 / DD −51%, 3/5 fold profitabel.
Gate `majority_folds_profitable` lolos sendirian; 4 gate lain gagal →
keputusan akhir `false`. Tepat seperti seharusnya: mesin tidak mengarang
kemenangan.

### 2.7 Jejak repair data

Manifest repair tercatat dan terverifikasi ulang: 731 baris stale
(volume-nol) dibuang, faktor resmi GAL/60 dan OMNI/75 diterapkan dengan
cutover **tanggal kalender** (bukan timestamp intraday — aturan lama sempat
memfabrikasi lonjakan +72x fiktif, tertangkap, diperbaiki, dan dikunci
regression test). Seam migrasi akhir: G −9,5%, NOM −2,5% (repricing riil),
POL −1,3%; duplikat sisa 0. Suite **39 test lulus**.

### 2.8 Struktur biaya midcap revisi M11B

Riset biaya (Novy-Marx & Velikov 2016: biaya eksekusi anomali mid-turnover
20–57 bps; ambang selamat turnover satu arah <50%/bulan ≈ 0,023/hari;
Binance USDⓈ-M taker 5 bps / maker 2 bps, diskon BNB −10%) menghasilkan
tier baru — kaki fee diratakan faktual, diferensiasi dipindah ke slippage:

| Tier (persentil vol) | Fee | Slippage | Total/unit |
|---|---|---|---|
| Top (≥p75) | 5 bps | 5 bps | **10** |
| Mid-high (p50–75) | 5 bps | 10 bps | **15** |
| Mid-low (p25–50) | 5 bps | 20 bps | **25** |
| Bottom (<p25) | 5 bps | 40 bps | **55** |

Ditambah: mode `slippage_mode="spread"` (`slippage = max(full spread
terukur, floor tier)`, fail-closed bila file hilang, fallback floor bila aset
tak ada di CSV), flag `turnover_flag` (`OVER` bila average turnover >0,15)
di tiap segmen validasi, dan tier + mode tercatat di `analysis_summary.json`.

Hasil re-run funded dengan struktur baru (71 aset, mode spread aktif):
semua signal tetap negatif (terbaik `resid_reversal_30`: −10,7%, Sharpe
+0,09; turnover 0,08–0,31 — tiga signal berflag `OVER`); walk-forward −12,1%,
Sharpe 0,02, DD −51%. Biaya yang lebih jujur tidak menyelamatkan strategi —
yang berubah hanya keyakinan bahwa penolakan gate bukan artefak asumsi murah.

## 3. Kelemahan / catatan yang tersisa (bukan blocker audit)

1. **Ketegangan gap-tolerant + futures — SELESAI.** Preflight kini mentriase gap tiga jenis (`classify_timestamp_gaps` di `lifecycle.py`): `migration_halt` (blok kosong berisi/berdampingan effective date resmi — fakta halt bursa, selalu warning, tidak pernah error), tepi listing/delisting (ditangani segmen lifecycle), dan `unexplained_interior` (satu-satunya yang bisa menggagalkan run futures; `--allow-gaps` kini hanya mentoleransi jenis ini, eksplorasi saja). Verifikasi di data nyata: 43 gap dataset funded (GUSDT:34, POLUSDT:9) semuanya halt migrasi kontinu (GAL halt 12 Jul 2024 → G jalan 15 Agu; MATIC halt 5 Sep → POL 13 Sep) — preflight ketat (`allow_gaps=False`) kini `valid: True` tanpa flag. Suite 67 test lulus.
2. **Sampel forced-exit kecil.** Mekanisme terbukti bekerja (1 event
   tercatat + regression test), tetapi jarang terpicu pada konfigurasi ini —
   keyakinan statistik atas jalur itu terbatas.
3. **Snapshot spread tunggal.** Valid untuk kalibrasi tier, belum mewakili
   pelebaran spread saat krisis (ditutup sebagian oleh stress 2×/4×).
4. **Survivorship bias — kini TERUKUR eksplisit (was 🔴).** Historical listing manifest (1028 simbol futures, 459 mati; validasi silang: LUNA mati 2022-05-13, G listing 2024-08-15, POL 2024-09-13 — semua cocok sejarah) dipasang ke lifecycle + preflight. Dataset funded 47-aset vs 934 simbol in-window: **365 dead-in-window tak tercakup** (upper bound termasuk tracker non-crypto; carrier material: FTT, SRM, ALPHA, BNX, REEF, WAVES, LOOM, KDA, REN...). Kontrol negatif lolos: LUNA ter-exclude dengan benar (pre-window), GAL/MATIC ter-exclude dengan benar (suksesor di dataset). Batasan sisa: tanggal listing/delisting adalah proksi file Vision pertama/terakhir (±hari vs pengumuman resmi), dan angka 365 adalah batas atas — namun arah bias kini tak terbantahkan: cross-section tanpa nama-nama itu terflatter secara konstruksi.

## 4. Vonis

| Aspek | Nilai |
|---|---|
| Preflight & validasi input | Baik — ketat, informatif, fail-fast |
| Canonical mapping & repair migrasi | Baik — terdokumentasi, ter-test, tanpa tebakan |
| Perhitungan PnL (harga + funding − biaya) | Baik — terdekomposisi, terverifikasi |
| Cost stress & capacity | Baik — bug lama mati, batas aktif |
| Struktur biaya & turnover flag | Baik — fee faktual 5 bps, slippage spread-calibrated, ambang turnover tercatat |
| Walk-forward & gate keputusan | Baik — bebas leakage, fail-closed |
| Spread check | Baik — data riil, asumsi konservatif |
| Kejujuran laporan (manifest, warning, note) | Baik — tidak ada angka siluman |

**Kesimpulan: sistem checker-nya berjalan baik dan layak dipakai sebagai
mesin audit.** `DEPLOYABLE False` pada run ini adalah bukti mesinnya bekerja
— ia menolak strategi yang memang merugi (−13,6% WF, funding −16,5rb pada
low_vol_30), bukan bukti mesinnya rusak. Perbaikan yang masih layak dilakukan
hanya penyelarasan teks warning gap-tolerant (poin 3.1) dan pengulangan spread
check berkala; tidak ada bug fungsional yang tersisa dari sisi audit ini.

## 5. Recheck engine terbaru (`reports/recheck_v2`, dataset funded 47-aset)

Sejak run di atas, lima kemampuan ditambahkan (commit `2be58c7`): asset
lifecycle engine, aturan eksekusi tunggal signal(t)→t+1, Deflated Sharpe Ratio,
capacity curve AUM, dan ensemble signal. Dataset
`data/midcap_2y_daily_2_funded.csv` (47 aset, 2024 → 2026-08, funding 100%)
dijalankan ulang penuh (`scripts/recheck_v2.py` + `recheck_v2b.py`).
Suite kini **65 test lulus**.

### 5.1 Hasil

- Walk-forward agregat: **+52%, Sharpe 0.85, 4/5 fold profit** (low_vol_14 menang 4 fold, carry_lowvol_z 1 fold) — terlihat bagus.
- **DSR (65 trials): 0.175** — null bar 0.92 > observed 0.85 → seleksi kemungkinan beruntung. Alpha dibunuh.
- Static OOS kontinu (low_vol_14): negatif. Ensemble top-3: **−13,5%, Sharpe −0,22** — member lain (momentum_7/30, vol_adj_carry, ...) tidak membawa alpha independen; diversifikasi naif hanya mengencerkan.
- Capacity (low_vol_14, full-period +11%, Sharpe 0,31, DD −40%): sensible di 10k (headroom 11x) dan 100k (headroom 1,1x — mepet); breach di 1M (59 breach) dan 10M (partisipasi maks 450% volume — mustahil dieksekusi).
- Risk violations tetap ada; WF drawdown −37%. **Vonis: DEPLOYABLE False.**
- Preflight atas CSV mentah: `valid: False` satu-satunya sebab kolom kerja `signal` belum ada (dibuat `canonicalize`/`build_signals`) — bukan cacat data. Lifecycle: 47 segmen terbentuk bersih, funding 100%, likuiditas hadir.

### 5.2 Penilaian subsistem baru

| Subsistem | Nilai |
|---|---|
| Lifecycle engine | Baik — 47 segmen, batas migrasi official, listing inferensi dilabel eksplisit; manifest CSV siap kurasi manual |
| Aturan eksekusi t+1 | Baik — menutup lubang lookahead nyata (signal custom/default lama terisi di bar observasi); 10 test lama gagal persis sesuai prediksi teori dan dihitung ulang manual |
| DSR | Baik — melakukan tugasnya: membunuh WF +52% yang tanpa koreksi terlihat deployable; reduksi ke PSR saat N=1 terverifikasi independen |
| Ensemble | Baik — dilaporkan diagnostik saja, tidak masuk vonis; hasil merahnya konsisten (member sampah), bukan artefak |
| Capacity curve | Baik — re-run aktual per AUM, bukan ekstrapolasi; headroom kuantitatif; menolak mengarang likuiditas |

### 5.3 Kesimpulan recheck
Mesin kini menemukan (low_vol_14 menang 4/5 fold), menguji (gates + DSR +
capacity + ensemble), dan membunuh alpha karena DD — bukan karena DSR. Recompute
dengan registry yang sudah diperbaiki mencatat `raw_n_trials=5`,
`valid_n_trials=5`, `effective_n_trials=1`, dan `DSR=1.0`. Jadi `DEPLOYABLE False`
berasal dari DD walk-forward/OOS sekitar −46%/−37%; sinyal bukan lagi kandidat
yang dibunuh oleh selection bias pada run ini. Jalur berikutnya adalah
vol-targeting/leverage reduction, bukan membuang low_vol_14 dan mencari signal
dari nol. Angka DSR fixture bukan typo; test `test_dd_reject_not_downgraded_by_high_dsr`
mengunci bahwa DSR tinggi tidak boleh menimpa rejection berbasis DD.

## 6. Skenario ideal: validasi sistem di data kotor (71-aset) — TERKUNCI

Dataset `midcap_2y_daily_2.csv` (71 aset, mentah) menjalani siklus ideal penuh:

1. **Detect:** preflight ketat menolak — `Duplicate (timestamp, asset)` (668 baris overlap OMNI→NOM) + `Unexplained interior gaps: 17`. Audit vs manifest resmi menangkap korupsi sunyi: histori GAL berlabel GUSDT ter-flag `trading_before_listing` (audit kini symbol-level; pencocokan canonical-level akan meloloskannya).
2. **Repair** (`scripts/repair_71.py` di atas primitif `crypto_checker/repair.py`): relabel 211 G + 531 NOM pra-cutover, drop 731 stale (MKR 357!), exclude kanonikal NOM (seam −30,2% pasca-faktor resmi tak terverifikasi).
3. **Verify:** `valid: True`, `errors: []`, 42 hari migration_halt, GRT/THETA 1-hari jadi warning toleransi.

Siklus ini dikunci sebagai regression test (`test_system_validation_dirty_detect_repair_verify`) — bukan sekadar run manual. Argumennya: 47-aset membuktikan engine bekerja pada data bersih; 71-aset membuktikan engine bekerja pada data kotor dan mengidentifikasi masalah nyata. Suite 75 test lulus.

## 7. Scoreboard stabilitas integrasi (L1–L5) — INTEGRATION_READY

| Level | Syarat | Hasil |
|---|---|---|
| L1 Functional | 100% test pass | 78/78 ✅ |
| L2 Deterministic | 2 run identik | 5 run identik lintas proses (2 determinism + 3 burn-in, SHA256 sama) ✅ |
| L3 Schema | schema v1 + contract test | `schema_version: 1` di 5 artefak + validator + tabel kontrak README ✅ |
| L4 Regression | golden fixture CI | fixture sintetis byte-identik lintas proses + hash SHA256; benchmark 47/71 via `benchmark/*/` (hash + regenerasi, tanpa data di git) ✅ |
| L5 Runtime | 100 small + 3 full | 100/100 (5 syarat per run, slowest 0,38s) + 3/3 identik ✅ |
| Artifact integrity | atomic write + parse validation | fsync + rename atomik di semua writer + crash test ✅ |

Temuan terpenting selama pembuktian: golden test lintas-proses menangkap nondeterminisme ULP nyata (urutan iterasi `set` mengikuti `PYTHONHASHSEED` per proses) yang lolos dari uji determinisme satu-proses — diperbaiki dengan urutan kanonik `sorted()` di semua agregat `core.py`, lalu dibuktikan stabil di dua `PYTHONHASHSEED` berbeda. Tanpa test ini, Quantara akan menerima angka yang goyang antar run.

Catatan presisi: CSV adalah round-trip lossy di level ULP (terukur maks 2,8e-14) — golden fixture dibangkitkan dari byte CSV yang di-commit (bukan dari memori), sehingga byte-equality menguji mesin murni, bukan presisi I/O.

## 8. Kontrak pra-integrasi Quantara (schema v1.1, aditif — tetap version 1)
Empat keputusan pra-integrasi, semua terimplementasi + ter-test:

1. **Contract freeze:** envelope `decision.json` (`status`, `decision`, `deployable`, `deployable_meaning`, `gates`, `metrics` 8-field, `capacity`, `risk`, `data_quality`, `warnings`, `errors`, `artifacts`) — populasi mengikuti tabel locked: SUCCESS→metrics populated/decision set; FAILED_VALIDATION/CHECKER_ERROR→metrics & capacity null, decision null, errors populated, `deployable` false. Aturan versi: v1.0→v1.1 aditif (version tetap 1); bump ke 2 hanya untuk rename/hapus/ubah-tipe/ubah-semantik.
2. **Satu entry point resmi:** `validate_csv()` di `crypto_checker/api.py` (+ `python -m crypto_checker.validate`, exit 0/2/1); `cli.py` lama di-guard `main()` agar import-safe tanpa perubahan perilaku. Quantara tidak memanggil internal.
3. **Tiga keadaan eksplisit:** SUCCESS+APPROVED/REJECTED (REJECTED = ditolak karena merit, bukan error) vs FAILED_VALIDATION (evaluasi tidak selesai) vs CHECKER_ERROR (tanpa klaim riset). `decision.json` selalu ditulis, bahkan di jalur gagal.
4. **Makna deployable dikunci di kontrak:** "Lolos validation criteria checker. BUKAN izin live trading / real money." Suite 82 test lulus.

## 9. Checker v2 (Fase 0–5): registry, DSR gray-zone, CPCV, ensemble pre-check, sqrt capacity, automated gate

- **Trial registry:** semua attempt tercatat (gagal/pendek memakai `ExclusionReason` enum, bukan teks bebas); DSR memakai effective-N hasil clustering korelasi (null bar turun vs raw-65 — diterima apa adanya).
- **DSR v2 fail-loud** (Pearson kurtosis; contoh spek 3.1 mengonfirmasi) + gray-zone 0.5–0.95 di gate.
- **CPCV 45 path low_vol_14:** 84% profitabel, Sharpe mean +0.88, positif di semua regime (tanpa concentration flag) — TETAPI max DD −46% train / −37% OOS melewati threshold → REJECTED via DD. Verdict konsisten, alasan diperkaya (bukan selection-bias melainkan drawdown).
- **Ensemble pre-check** ter-wire (BLOCK folds tidak diagregat); **sqrt-impact overlay** + ADV 90d + warning linear; hard gate partisipasi utuh.
- **Gate otomatis** + mapping envelope (`review_required`, blok `gate_decision` bersarang, `decision=null` saat FLAG). Fixture `known_rejected_cases.json` mengunci kedua kasus REJECTED sebagai unit test logika gate.
