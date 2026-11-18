# Suit Jawa vs AI

Proyek computer vision dan permainan webcam untuk Suit Jawa melawan AI:

- **Gajah** — jempol
- **Orang** — telunjuk
- **Semut** — kelingking

Pipeline memisahkan utilitas sistem, ekstraksi fitur, model classifier,
antarmuka gestur, AI, game engine, dan pencatatan hasil.

## Struktur

```text
data/
  dataset_suit_jawa.csv     Fitur landmark, ID anonim, dan label (65 kolom)
  collection_metadata.csv   Log tangan, jarak, cahaya, dan waktu sampel
models/
  suit_jawa_classifier.joblib   Artifact classifier lokal
  evaluation_fase_1_4.md        Laporan evaluasi model
src/
  features/
    landmark_normalizer.py  Ekstraksi dan normalisasi 63 fitur tangan
  vision/
    benchmark_camera.py     Benchmark webcam, MediaPipe, dan FPS
    gesture_interface.py    Prediksi gestur dari webcam
  game/
    game_engine.py           FSM permainan
    ai_predictor.py          Prediksi pilihan pemain
    ai_counter.py            Counter AI
    game_logger.py           Log anonim append-only
  utils/
    system_info.py          Pembacaan spesifikasi perangkat
    csv_validator.py        Validasi round-trip tulis/baca CSV
    dataset_audit.py        Audit 11 indikator dan keseimbangan kelas
record_data.py              Perekam dataset landmark interaktif
main.py                     Entry point permainan
game_logs.csv               Riwayat ronde lintas sesi
tests/                      Unit dan integration tests
requirements.txt             Dependensi runtime
.gitignore                  Cache, environment, dan data sensitif lokal
dataset_report.md           Laporan otomatis audit dataset
benchmark_vision.md         Laporan benchmark yang dihasilkan skrip
```

## Privasi data dan version control

CSV landmark berisi fitur biometrik tangan yang dikaitkan dengan ID pseudonim
dan metadata waktu. Model merupakan turunan data tersebut, sedangkan
`game_logs.csv` berisi riwayat pilihan ronde. Karena itu `.gitignore`
mengecualikan CSV di `data/`, artifact model, dan log permainan dari staging
Git. Simpan file tersebut secara lokal; jangan membagikannya tanpa persetujuan
peserta dan peninjauan privasi yang sesuai.

## Persiapan

Disarankan menggunakan Python 3.10–3.12. Buat dan aktifkan virtual environment,
lalu pasang dependensi:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Gunakan Python dari `.venv` untuk memasang dependensi dan menjalankan benchmark
atau perekam dataset. Jika tidak mengaktifkan environment, perintah `python`
dapat menunjuk ke instalasi Python lain yang belum memiliki MediaPipe.

## Fase 1.1 — Persiapan dan Benchmark Laptop

- [x] Menyiapkan struktur proyek modular dan placeholder dataset/model.
- [x] Menyediakan pembacaan CPU, jumlah core, RAM, GPU, dan OS.
- [x] Menyediakan skrip benchmark webcam, MediaPipe Hands, dan FPS.
- [x] Menghasilkan laporan spesifikasi tanpa membuka webcam.
- [x] Pasang dependensi Python di environment proyek.
- [x] Jalankan benchmark webcam dan verifikasi izin/perangkat kamera.

Pengujian webcam fisik terbaru mencatat resolusi 640x480, **23.83 FPS**
selama 30.34 detik, dan deteksi tangan pada 41 dari 723 frame. Hasil dapat
berubah sesuai kamera dan beban perangkat; lihat `benchmark_vision.md` untuk
angka lengkap.

### Menjalankan benchmark

Catat spesifikasi saja tanpa mengaktifkan webcam:

```powershell
.\.venv\Scripts\python.exe -m src.vision.benchmark_camera --system-only
```

Jalankan uji webcam selama 30 detik (tekan `Q` untuk menghentikan lebih awal):

```powershell
.\.venv\Scripts\python.exe -m src.vision.benchmark_camera
```

Atur durasi atau indeks kamera bila diperlukan:

```powershell
.\.venv\Scripts\python.exe -m src.vision.benchmark_camera --duration 60 --camera-index 1
```

Gunakan `--no-display` untuk benchmark tanpa jendela preview. Laporan terbaru
ditulis ke `benchmark_vision.md`; opsi `--report` dapat mengubah lokasinya.

## Fase 1.2 — Perekaman Dataset

- [x] Menambahkan ekstraksi landmark tangan menjadi 63 fitur.
- [x] Menormalisasi fitur terhadap posisi pergelangan dan skala tangan.
- [x] Menambahkan perekam webcam dengan pemilihan label dan penyimpanan CSV.
- [x] Memvalidasi round-trip CSV tanpa menambahkan baris uji ke dataset.
- [x] Sampel gestur Gajah, Orang, dan Semut tersedia di CSV.

55 sampel pertama merupakan data legacy tanpa ID peserta dan metadata. Dataset
saat ini memiliki 564 baris: 55 legacy dan 509 sampel dari dua peserta
pseudonim. Sampel legacy tetap dipertahankan, tetapi tidak dihitung sebagai
sukarelawan terverifikasi.

## Fase 1.3 — Pengumpulan dan Audit Dataset

- [x] Menambahkan ID peserta anonim dan persetujuan eksplisit sebelum merekam.
- [x] Menambahkan label `unknown` serta HUD pemilihan kondisi pengambilan.
- [x] Menyimpan metadata tangan, jarak, pencahayaan, dan waktu per sampel.
- [x] Menyiapkan audit otomatis 11 indikator dan ringkasan keseimbangan kelas.
- [ ] Kumpulkan data dengan minimal 5 sukarelawan (ID berbeda).
- [ ] Lengkapi empat label, tiga gestur utama untuk tiap peserta, kedua tangan,
      tiga jarak, dan tiga kondisi pencahayaan.
- [ ] Luluskan audit dan periksa [dataset_report.md](./dataset_report.md).

Untuk perekaman gunakan ID anonim `P001`-`P999` dan pastikan peserta telah
memberi persetujuan. Tekan `1` untuk **Gajah** (jempol), `2` untuk **Orang**
(telunjuk), `3` untuk **Semut** (kelingking), atau `4` untuk `unknown`.
Pilih jarak dengan `J/K/L` (dekat/sedang/jauh), pencahayaan dengan `A/S/D`
(redup/normal/terang), lalu tekan `SPACE` untuk menyimpan dan `Q` untuk selesai.

```powershell
.\.venv\Scripts\python.exe record_data.py --participant-id P001 --consent-confirmed
```

Setiap baris dataset memiliki tepat 65 kolom: `participant_id`, `label`, dan
`x0,y0,z0,...,x20,y20,z20`. Metadata per baris disimpan terpisah di
`data/collection_metadata.csv`; kolom `dataset_row` menghubungkan metadata
dengan urutan baris dataset. Perekam hanya menyimpan landmark dan metadata,
bukan frame atau video. `--camera-index 1`, `--output path\dataset.csv`, dan
`--metadata path\metadata.csv` dapat digunakan untuk mengubah perangkat/lokasi.

Jalankan audit kapan saja, termasuk saat pengumpulan belum lengkap:

```powershell
.\.venv\Scripts\python.exe -m src.utils.dataset_audit
```

Laporan menetapkan keseimbangan kelas utama Gajah/Orang/Semut pada rasio
maksimum 1.5:1 (dapat diubah dengan `--balance-ratio`). Label `unknown`
dilaporkan terpisah. Data lama fase 1.2 dimigrasikan dengan ID
`LEGACY_UNVERIFIED`; isinya dipertahankan, tetapi tidak dihitung sebagai
sukarelawan atau sebagai sampel dengan metadata yang terverifikasi.

---

### Target pengumpulan untuk lima sukarelawan

Gunakan ID pseudonim `P001`–`P005`; jangan gunakan nama. Target 40 sampel
untuk setiap label per peserta menghasilkan 160 sampel per peserta. Untuk
setiap label, sebar sampel ke kombinasi tangan kiri/kanan, jarak dekat/sedang/
jauh, dan pencahayaan redup/normal/terang. Usahakan sedikitnya dua sampel per
kombinasi (36 sampel), lalu gunakan empat sampel sisanya untuk variasi gestur
atau kondisi yang paling sering keliru. Pada label `unknown`, rekam kepalan,
telapak terbuka, jari acak, dan transisi gestur.

Audit data terakhir lulus **6 dari 11 indikator**. Rincian terbaru tersedia
di [dataset_report.md](./dataset_report.md); evaluasi classifier tersedia di
[evaluation_fase_1_4.md](./models/evaluation_fase_1_4.md). Keduanya menegaskan
bahwa data belum cukup mewakili variasi pengguna.

## Fase 1.4 — Training dan Demo Offline

Trainer mengecualikan baris `LEGACY_UNVERIFIED`, membagi data berdasarkan
`participant_id` (bukan per baris), dan hanya menerima split yang memiliki
keempat label pada Train Set dan Test Set. Output terminal menampilkan daftar
peserta di kedua set serta confusion matrix berurutan
`Gajah`, `Orang`, `Semut`, `unknown`. Pastikan daftar peserta tidak tumpang
tindih dan periksa baris/kolom `unknown`, terutama setelah menambahkan data
gestur netral, mengepal, atau acak.

```powershell
.\.venv\Scripts\python.exe train_classifier.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe demo_vision.py --mode offline
```

Model disimpan lokal di `models/suit_jawa_classifier.joblib`. Demo memakai
MediaPipe dan model lokal tanpa layanan jaringan; saat jendela terbuka, uji
bergantian ketiga gestur utama dan gestur netral/mengepal, termasuk saat
perpindahan antar gestur. Koneksi internet tidak diperlukan.

Evaluasi lintas peserta memerlukan minimal dua peserta yang masing-masing
memiliki seluruh label; hasil awal dengan kurang dari lima peserta ditandai
sebagai belum cukup mewakili variasi pengguna. Kumpulkan data dari lebih
banyak peserta sebelum menganggap hasil uji sebagai kinerja umum.

## Tahap 2.1 — Integrasi Gesture Classifier

`GestureInterface` memiliki kamera dan pipeline MediaPipe serta mengembalikan
`GesturePrediction(label, confidence, is_valid, hand_detected)`; Game Engine tidak perlu
mengolah landmark. `is_valid` hanya bernilai benar untuk Gajah/Orang/Semut
dengan confidence minimal 0.60. Tidak terdeteksinya tangan, prediksi `unknown`,
atau confidence di bawah ambang menghasilkan label `unknown` dan
`is_valid=False`. Ambang dapat diubah lewat argumen `confidence_threshold`.

```python
from src import GestureInterface

with GestureInterface() as gesture:
    prediction = gesture.get_prediction()
    if prediction.is_valid:
        print(prediction.label, prediction.confidence)
```

Jalankan pengujian interface tanpa webcam:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gesture_interface -v
```

## Tahap 2.2 — Game Engine

`SuitJawaEngine` tidak menggambar UI. Ia menerima `GestureInterface` opsional
atau `GesturePrediction` per panggilan `update()`. State-nya adalah
`READY` → `COUNTDOWN` → `COUNTER_REVEAL` → `COLLECTING_GESTURE` →
`ROUND_RESULT` → `FINISHED`.
Default-nya 10 ronde, countdown 3 detik, batas input 5 detik, stabilisasi
gestur 0,35 detik, reveal counter 1 detik, dan tampilan hasil 1,5 detik.
Nilai-nilai tersebut dapat diubah pada constructor.

Prediksi `unknown`, tangan yang tidak terdeteksi, dan gestur tidak valid
diabaikan selama batas input; perubahan gestur mereset timer stabilisasi.
Jika tak ada gestur valid yang stabil sebelum timeout, ronde dicatat sebagai
`Gestur tidak stabil` tanpa mengubah skor.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_game_engine -v
```

## Tahap 2.3 — AI Predictor dan Counter

`AIPredictor` menghasilkan distribusi probabilitas dengan prior frekuensi
bersmoothing dan transisi Markov orde 1 (bobot Markov default 0,7). Pilihan
dengan probabilitas tertinggi diprediksi; jika seri, predictor memilih secara
acak hanya di antara pilihan yang seri. `AICounter` memilih langkah yang
mengalahkan prediksi tersebut.

Saat memasuki `COUNTER_REVEAL`, engine menghitung prediksi dari gestur pemain
pada ronde valid sebelumnya dan menyimpan counter AI. Pesan `engine.message`,
`engine.forecast`, serta `engine.planned_ai_gesture` tersedia untuk UI selama
fase reveal. `GestureInterface` baru dipanggil pada fase
`COLLECTING_GESTURE`, sesudah fase reveal selesai, sehingga pilihan aktual
ronde itu tidak digunakan untuk menentukan counternya sendiri. Ronde timeout
tanpa gestur valid tidak masuk ke riwayat predictor.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_ai_predictor -v
```

## Tahap 2.4 — Game Logs

`GameLogger` menambahkan satu baris untuk setiap ronde yang selesai, termasuk
ronde timeout. File [game_logs.csv](./game_logs.csv) dibuka dalam mode append;
baris lama tidak ditimpa. Setiap pemanggilan `engine.start()` mendapat
`session_id` UUID4 baru. Kolom log hanya berisi ID sesi anonim, nomor ronde,
gestur pemain, prediksi dan counter AI, pemenang, serta skor. Data landmark,
confidence, nama, informasi perangkat, dan identitas lain tidak dicatat.
Untuk ronde timeout, gestur pemain dan pemenang disimpan kosong/`none`,
sedangkan prediksi serta counter yang telah disiapkan tetap dicatat.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_game_logger -v
```

## Tahap 2.5 — Integrasi dan Pengujian

Jalankan permainan dari root proyek. Model, MediaPipe, dan classifier berjalan
lokal; mode offline tidak memerlukan koneksi internet:

```powershell
.\.venv\Scripts\python.exe main.py --mode offline --camera-index 0 --rounds 10
```

Gunakan `--model`, `--camera-index`, atau `--log-file` untuk mengganti artifact,
perangkat kamera, atau lokasi CSV. Jika kamera gagal dibuka atau terputus saat
permainan, aplikasi menampilkan error dan menutup resource kamera. Hasil
simulasi untuk kamera terputus, gestur tidak stabil, mode offline, sesi penuh,
dan beberapa sesi dapat dijalankan tanpa webcam:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_integration_e2e -v
```
