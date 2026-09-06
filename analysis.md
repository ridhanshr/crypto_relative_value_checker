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

1. **Ketegangan gap-tolerant + futures.** Run ini memakai `--allow-gaps`
   bersama `--require-funding`, padahal teks warning preflight menyebut mode
   toleran-gap tidak valid untuk deployment futures. Praktiknya aman untuk
   *riset* (59 gap = halt/delisting nyata, ditangani forced exit, semuanya
   tercatat), tetapi teks warning vs perilaku perlu diselaraskan: bedakan
   "gap halt bursa yang terdokumentasi" vs "gap data yang tak terjelaskan".
2. **Sampel forced-exit kecil.** Mekanisme terbukti bekerja (1 event
   tercatat + regression test), tetapi jarang terpicu pada konfigurasi ini —
   keyakinan statistik atas jalur itu terbatas.
3. **Snapshot spread tunggal.** Valid untuk kalibrasi tier, belum mewakili
   pelebaran spread saat krisis (ditutup sebagian oleh stress 2×/4×).
4. **Survivorship bias** tetap dicatat belum terukur — batasan data, bukan
   kegagalan mesin.

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
