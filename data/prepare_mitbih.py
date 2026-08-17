"""
Convert raw MIT-BIH Arrhythmia Database WFDB records into fixed-length
beat windows + AAMI 5-class labels (N, S, V, F, Q), saved as X.npy /
y.npy for src/training/train_ta_cnn.py.

Prerequisite: download MIT-BIH locally first (no PhysioNet credentials
needed -- it is fully open):

    wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/ \
        -P ./raw_mitbih/

Or via the wfdb package's built-in downloader:

    python -c "import wfdb; wfdb.dl_database('mitdb', './raw_mitbih')"

Usage:
    python prepare_mitbih.py --raw-dir ./raw_mitbih --out-dir ./processed_mitbih
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import wfdb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.preprocessing.filters import butterworth_bandpass

# AAMI EC57 grouping of MIT-BIH beat annotation symbols
AAMI_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,          # Normal
    "A": 1, "a": 1, "J": 1, "S": 1,                   # Supraventricular
    "V": 2, "E": 2,                                   # Ventricular
    "F": 3,                                           # Fusion
    "/": 4, "f": 4, "Q": 4,                           # Unknown/paced
}

WINDOW_SAMPLES = 250  # ~0.7s at 360 Hz MIT-BIH sampling rate, centered on R-peak


def process_record(record_path: str):
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")
    signal = record.p_signal[:, 0]  # first lead (typically MLII)

    filtered = butterworth_bandpass(signal, fs=record.fs, low_hz=0.5,
                                     high_hz=min(40.0, record.fs / 2 - 1))

    windows, labels = [], []
    half = WINDOW_SAMPLES // 2
    for sample_idx, symbol in zip(ann.sample, ann.symbol):
        if symbol not in AAMI_MAP:
            continue
        if sample_idx - half < 0 or sample_idx + half > len(filtered):
            continue
        windows.append(filtered[sample_idx - half: sample_idx + half])
        labels.append(AAMI_MAP[symbol])

    return np.array(windows), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True,
                         help="Directory containing MIT-BIH .dat/.hea/.atr files")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record_names = sorted({p.stem for p in raw_dir.glob("*.hea")})
    if not record_names:
        raise FileNotFoundError(
            f"No .hea files found in {raw_dir}. Download MIT-BIH first "
            f"(see module docstring)."
        )

    all_windows, all_labels, all_record_ids = [], [], []
    for name in record_names:
        try:
            windows, labels = process_record(str(raw_dir / name))
        except Exception as exc:
            print(f"Skipping record {name}: {exc}")
            continue
        all_windows.append(windows)
        all_labels.append(labels)
        all_record_ids.extend([name] * len(labels))
        print(f"Record {name}: {len(labels)} labeled beats")

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    record_ids = np.array(all_record_ids)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "record_ids.npy", record_ids)  # for patient-independent splitting

    print(f"\nSaved {len(y)} beat windows to {out_dir}")
    print("Class distribution (AAMI N/S/V/F/Q):",
          {c: int(np.sum(y == c)) for c in range(5)})
    print("\nIMPORTANT: use record_ids.npy to build a PATIENT-INDEPENDENT "
          "train/val/test split (no record's beats should appear in both "
          "train and val) -- see data/patient_split.py. A random beat-level "
          "split will leak information and inflate accuracy.")


if __name__ == "__main__":
    main()
