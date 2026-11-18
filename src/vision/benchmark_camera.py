"""Benchmark webcam capture and MediaPipe hand detection."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any

from src.utils.system_info import get_system_info


DEFAULT_REPORT = Path(__file__).resolve().parents[2] / "benchmark_vision.md"


def run_camera_benchmark(
    camera_index: int = 0,
    duration: float = 30.0,
    show_window: bool = True,
) -> dict[str, Any]:
    """Capture frames and measure MediaPipe hand-detection throughput."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            f"OpenCV belum tersedia untuk Python yang sedang dipakai "
            f"({sys.executable}). Aktifkan .venv atau jalankan "
            "`.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
        ) from error
    try:
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            f"MediaPipe belum tersedia untuk Python yang sedang dipakai "
            f"({sys.executable}). Aktifkan .venv atau jalankan "
            "`.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
        ) from error

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Webcam dengan indeks {camera_index} tidak dapat dibuka. "
            "Periksa koneksi, izin kamera, dan indeks perangkat."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0
    frames_with_hands = 0
    detected_hands = 0
    hands_api = mp.solutions.hands

    try:
        with hands_api.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            started_at = time.perf_counter()
            deadline = started_at + duration
            while time.perf_counter() < deadline:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Gagal membaca frame dari webcam.")

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)
                frame_count += 1
                hand_count = len(result.multi_hand_landmarks or [])
                if hand_count:
                    frames_with_hands += 1
                    detected_hands += hand_count

                if show_window:
                    if result.multi_hand_landmarks:
                        for landmarks in result.multi_hand_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(
                                frame,
                                landmarks,
                                hands_api.HAND_CONNECTIONS,
                            )
                    cv2.imshow("Suit Jawa vs AI - Benchmark (tekan Q untuk berhenti)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        capture.release()
        if show_window:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started_at
    if frame_count == 0:
        raise RuntimeError("Tidak ada frame yang berhasil diproses.")

    return {
        "status": "Selesai",
        "camera_index": camera_index,
        "resolution": f"{frame_width}x{frame_height}",
        "duration_seconds": round(elapsed, 2),
        "frames_processed": frame_count,
        "processing_fps": round(frame_count / elapsed, 2),
        "frames_with_hands": frames_with_hands,
        "hands_detected": detected_hands,
    }


def write_report(
    report_path: Path,
    system_info: dict[str, Any],
    benchmark: dict[str, Any] | None,
    error: str | None = None,
) -> None:
    """Write the machine details and the latest benchmark outcome to Markdown."""
    if benchmark is None:
        camera_status = "Belum dijalankan (mode `--system-only`)."
        benchmark_rows = ["| Hasil | Belum tersedia |"]
    else:
        camera_status = benchmark["status"]
        benchmark_rows = [
            f"| Indeks kamera | {benchmark['camera_index']} |",
            f"| Resolusi aktual | {benchmark['resolution']} |",
            f"| Durasi | {benchmark['duration_seconds']} detik |",
            f"| Frame diproses | {benchmark['frames_processed']} |",
            f"| FPS pemrosesan | {benchmark['processing_fps']} |",
            f"| Frame dengan tangan | {benchmark['frames_with_hands']} |",
            f"| Total tangan terdeteksi | {benchmark['hands_detected']} |",
        ]
    if error:
        camera_status = "Gagal"
        benchmark_rows.append(f"| Error | {error} |")

    gpu_names = system_info["gpu"] or ["Tidak terdeteksi"]
    report = [
        "# Laporan Benchmark Computer Vision",
        "",
        f"- Waktu pembuatan: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Sistem operasi: {system_info['os']}",
        f"- Prosesor: {system_info['cpu']}",
        f"- Core CPU: {system_info['physical_cores']} fisik / "
        f"{system_info['logical_cores']} logis",
        f"- RAM: {system_info['ram_gb']} GB",
        f"- GPU: {', '.join(gpu_names)}",
        "",
        "## Benchmark webcam dan MediaPipe",
        "",
        f"- Status: {camera_status}",
        "",
        "| Metrik | Nilai |",
        "| --- | --- |",
        *benchmark_rows,
        "",
        "FPS adalah jumlah frame yang berhasil diproses MediaPipe per detik "
        "selama pengujian; hasil aktual dipengaruhi kamera, resolusi, dan beban sistem.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0, help="Indeks webcam (default: 0)")
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Durasi benchmark dalam detik (default: 30)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Lokasi laporan Markdown (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Jalankan tanpa jendela preview",
    )
    parser.add_argument(
        "--system-only",
        action="store_true",
        help="Catat spesifikasi sistem tanpa membuka webcam",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration harus lebih besar dari 0")

    system_info = get_system_info()
    if args.system_only:
        write_report(args.report, system_info, None)
        print(f"Laporan spesifikasi sistem ditulis ke {args.report}")
        return 0

    try:
        benchmark = run_camera_benchmark(
            camera_index=args.camera_index,
            duration=args.duration,
            show_window=not args.no_display,
        )
    except RuntimeError as error:
        write_report(args.report, system_info, None, str(error))
        print(f"Benchmark gagal: {error}")
        print(f"Detail dan spesifikasi sistem dicatat di {args.report}")
        return 1

    write_report(args.report, system_info, benchmark)
    print(f"Benchmark selesai: {benchmark['processing_fps']} FPS")
    print(f"Laporan ditulis ke {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
