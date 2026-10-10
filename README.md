# TSP Metaheuristic: Simulated Annealing vs Tabu Search

Aplikasi Streamlit untuk TSP dengan perbandingan **Simulated Annealing (SA)** dan **Tabu Search (TS)**.

## Struktur
- `app/app.py` : interface Streamlit (tema pink-kuning di `.streamlit/config.toml`)
- `src/tsp_core.py` : jarak, NN/NI/CI/FI/AI, 2-opt, Swap, 3-opt, SA, TS, ILS
- `data/data1.xlsx` : dataset (Case I = 25, II = 40, III = 68 kota; sheet `Sheet7`)
- `docs/` : tempat teori, metodologi, case study, user guide (isi untuk paper/GitHub)

## Instalasi & Menjalankan
```
pip install -r requirements.txt
streamlit run app/app.py
```
## Cara pakai
1. Sidebar 1: pilih input (acak / manual / Excel-CSV / dataset contoh), atur kolom City, X, Y bila upload.
2. Sidebar 2: pilih metric, heuristik awal (NN/NI/CI/FI/AI) atau rute manual (mis. `1,4,2,5,3,6`).
3. Sidebar 3-6: 2-opt awal, parameter SA & TS, ILS, multi-run.
4. Cek *Optimization Summary*, klik **RUN OPTIMIZATION**.
5. Baca hasil: ringkasan, rute, konvergensi, detail rute (CSV), log (CSV), distance matrix (CSV), analisis, PNG.
6. **Ilustrasi Tabu Search**: input "Dataset contoh", Case I, N = 6, lalu lihat *Tabu Search Explorer*.

## Catatan implementasi
- Rute = tur tertutup; setiap kota tepat sekali. Batas kota 200.
- Neighborhood: Swap, 2-opt (balik segmen), 3-opt (tukar dua segmen berurutan).
- SA: Metropolis `exp(-Δ/T)`, `T ← α·T` setiap `iterasi per temperatur`; berhenti jika T < Tmin atau iterasi habis.
- TS: tabu list FIFO berukuran *tabu size* berisi pasangan kota yang ditukar; aspiration = tabu boleh dipilih jika lebih baik dari best.
- ILS: 2-opt + perturbasi double-bridge, terima jika tidak lebih buruk.

## Fitur interaktif
- **Klik kanvas X-Y**: klik area kosong untuk menambah kota (dibulatkan ke grid 50x50), mode Hapus, Undo, Reset.
- **SA vs TS berdampingan**: edge yang sama = garis tipis, edge unik = garis tebal berwarna.
- **Kompetisi**: klik kota sesuai urutanmu, lalu bandingkan dengan Initial, SA, TS (dan ILS).

## Alur UI (4 langkah)
1. **Data kota**: pilih metode input di sidebar (klik kanvas, acak, manual, Excel/CSV, dataset contoh).
2. **Pengaturan algoritma**: expander "Pengaturan lanjutan" berisi tab Jarak & Solusi Awal, Local Search, SA, Tabu Search, ILS (jika dicentang), Multi-run & Seed.
3. **Ringkasan konfigurasi**: kartu Problem Overview, Optimization Configuration (chips parameter), dan Initial Solution. Berstatus *belum dijalankan* sampai tombol Run ditekan.
4. **Hasil optimasi**: tab Ringkasan, Rute, SA vs TS, Konvergensi, Detail & Log, Analisis, Kompetisi (+ Tabu Explorer, Multi-run).
