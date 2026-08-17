"""
Patient-independent (record-independent) train/val/test split.

The manuscript claims a "patient-independent evaluation protocol" --
this is what actually enforces that claim: it splits by record/patient
ID first, THEN assigns beats, so no patient's data leaks across
splits. Using sklearn's plain train_test_split on beat windows
directly (without this) is a common and serious leakage bug that
inflates reported accuracy.

Usage:
    python patient_split.py --data-dir ./processed_mitbih \
        --out-dir ./processed_mitbih_split --test-size 0.2 --val-size 0.1
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                         help="Directory with X.npy, y.npy, record_ids.npy")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    record_ids = np.load(data_dir / "record_ids.npy")

    unique_records = np.unique(record_ids)
    train_val_recs, test_recs = train_test_split(
        unique_records, test_size=args.test_size, random_state=args.seed)
    train_recs, val_recs = train_test_split(
        train_val_recs, test_size=args.val_size / (1 - args.test_size),
        random_state=args.seed)

    def subset(recs):
        mask = np.isin(record_ids, recs)
        return X[mask], y[mask]

    splits = {"train": subset(train_recs), "val": subset(val_recs),
              "test": subset(test_recs)}

    out_dir = Path(args.out_dir)
    for split_name, (Xs, ys) in splits.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "X.npy", Xs)
        np.save(split_dir / "y.npy", ys)
        print(f"{split_name}: {len(ys)} beats from "
              f"{len(train_recs) if split_name == 'train' else len(val_recs) if split_name == 'val' else len(test_recs)} records")

    # Sanity check: verify no record appears in more than one split
    assert not set(train_recs) & set(val_recs)
    assert not set(train_recs) & set(test_recs)
    assert not set(val_recs) & set(test_recs)
    print("\nVerified: no patient/record overlap between splits.")


if __name__ == "__main__":
    main()
