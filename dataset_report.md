# Laporan Audit Dataset Suit Jawa vs AI

- Dibuat: 2026-11-18T11:51:42+00:00
- Dataset: `D:\PROJECT KULIAH\PKL\suit-jawa-ai\data\dataset_suit_jawa.csv`
- Metadata: `D:\PROJECT KULIAH\PKL\suit-jawa-ai\data\collection_metadata.csv`
- Status keseluruhan: **BELUM LULUS** (6/11 indikator)

Audit hanya menghitung ID sukarelawan anonim berbentuk `P001`-`P999`. 55 sampel lama ditandai `LEGACY_UNVERIFIED` karena identitas dan kondisi pengumpulannya tidak tersedia; sampel tersebut dipertahankan, tetapi tidak dihitung sebagai sukarelawan terverifikasi.

Keseimbangan dihitung hanya untuk Gajah/Orang/Semut; `unknown` adalah kelas tambahan untuk gestur di luar tiga kelas permainan. Ambang keseimbangan default adalah rasio kelas terbesar terhadap terkecil maksimal 1.50:1.

## 11 indikator

| # | Indikator | Status | Hasil |
| --- | --- | --- | --- |
| 1 | 1. Skema CSV dataset tepat 65 kolom | LULUS | 65 kolom ditemukan; 65 diwajibkan. |
| 2 | 2. Dataset berisi baris dengan struktur lengkap | LULUS | 564 baris; 0 baris memiliki kolom kosong/berlebih; 0 ID peserta tidak valid. |
| 3 | 3. Empat label valid (Gajah, Orang, Semut, unknown) terwakili | LULUS | Gajah: 150, Orang: 113, Semut: 99, unknown: 202 |
| 4 | 4. Semua 63 fitur numerik dan finite | LULUS | Seluruh nilai fitur valid. |
| 5 | 5. Minimal 5 sukarelawan anonim terverifikasi | BELUM LULUS | 2 ID unik memenuhi format P001-P999; 55 baris legacy dikecualikan. |
| 6 | 6. Setiap sukarelawan memiliki tiga gestur utama | LULUS | 2/2 peserta memiliki Gajah, Orang, dan Semut. |
| 7 | 7. Metadata lengkap dan cocok untuk setiap sampel terverifikasi | BELUM LULUS | 439 metadata untuk 509 sampel terverifikasi; data legacy tidak dihitung. |
| 8 | 8. Variasi tangan kiri dan kanan tercatat | LULUS | Left: 56, Right: 383 |
| 9 | 9. Variasi jarak dekat, sedang, dan jauh tercatat | BELUM LULUS | far: 0, medium: 248, near: 191 |
| 10 | 10. Variasi pencahayaan redup, normal, dan terang tercatat | BELUM LULUS | bright: 0, low: 237, normal: 202 |
| 11 | 11. Keseimbangan tiga kelas utama (rasio maksimum 1.50:1) | BELUM LULUS | Rasio 1.61:1; Gajah: 135, Orang: 88, Semut: 84 |

## Distribusi label

| Label | Semua baris | Peserta terverifikasi |
| --- | ---: | ---: |
| Gajah | 150 | 135 |
| Orang | 113 | 88 |
| Semut | 99 | 84 |
| unknown | 202 | 202 |

## Pemeriksaan tambahan

- Duplikasi vektor fitur persis pada peserta terverifikasi: 0.
- Baris legacy tanpa metadata yang dapat diverifikasi: 55.

## Batasan dan privasi

Perekam menyimpan landmark dan metadata kondisi, bukan frame/video. Gunakan ID pseudonim, dapatkan persetujuan peserta, dan jangan menambahkan nama, email, atau pengenal langsung ke file dataset.
