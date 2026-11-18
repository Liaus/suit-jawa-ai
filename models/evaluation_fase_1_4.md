# Evaluasi Model — Fase 1.4

Evaluasi ini dijalankan pada 18 November 2026 menggunakan
`train_classifier.py`, `test_size=0.25`, dan `random_state=42`. Model dilatih
ulang untuk pengukuran ke file sementara; artifact
`suit_jawa_classifier.joblib` yang ada tidak ditimpa.

## Data dan metode

- Dataset: 564 baris; 55 baris legacy dikecualikan.
- Sampel terverifikasi: 509 dari 2 peserta pseudonim.
- Split dilakukan berdasarkan peserta, bukan baris.
- Data train: 288 sampel; data test: 221 sampel.
- Classifier: SVC dengan `class_weight="balanced"` dan probabilitas
  terkalibrasi.

## Hasil

- Accuracy: **0.56**
- Macro F1: **0.47**
- Weighted F1: **0.55**

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Gajah | 0.29 | 0.88 | 0.43 | 17 |
| Orang | 0.62 | 0.43 | 0.51 | 35 |
| Semut | 0.44 | 0.19 | 0.27 | 42 |
| unknown | 0.67 | 0.67 | 0.67 | 127 |

Confusion matrix (baris = label aktual, kolom = prediksi):

| Aktual \ Prediksi | Gajah | Orang | Semut | unknown |
| --- | ---: | ---: | ---: | ---: |
| Gajah | 15 | 0 | 0 | 2 |
| Orang | 10 | 15 | 0 | 10 |
| Semut | 4 | 0 | 8 | 30 |
| unknown | 23 | 9 | 10 | 85 |

## Batasan dan tindak lanjut

Hasil ini belum cukup mewakili variasi pengguna karena hanya ada dua peserta
terverifikasi. Recall Semut yang rendah juga menunjukkan bahwa sebagian besar
gestur Semut pada test set belum dikenali dengan baik. Audit dataset terakhir
lulus 6 dari 11 indikator; metadata belum mencakup seluruh sampel, belum ada
variasi jarak jauh maupun pencahayaan terang, dan keseimbangan kelas utama
belum memenuhi ambang. Kumpulkan data tambahan dari peserta yang menyetujui
perekaman sebelum menggunakan hasil ini sebagai klaim kinerja umum.
