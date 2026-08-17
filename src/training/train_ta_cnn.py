"""
Training loop for TA-CNN on MIT-BIH Arrhythmia Database.

REQUIRES REAL DATA: point --data-dir at a local copy of MIT-BIH
(WFDB format, freely downloadable from PhysioNet without
credentialing -- see data/README.md for instructions). This script
does not fabricate or assume any accuracy number; it prints exactly
what the model achieves on your data.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.ta_cnn import TACNN
from src.preprocessing.filters import butterworth_bandpass, normalize_signal


class BeatWindowDataset(Dataset):
    """Loads fixed-length beat windows + AAMI labels from a directory
    of preprocessed .npy files. Use data/prepare_mitbih.py to build
    this directory from raw WFDB records (requires the `wfdb` package
    and a local copy of MIT-BIH).
    """

    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        x_path, y_path = data_dir / "X.npy", data_dir / "y.npy"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Expected {x_path} and {y_path}. Run "
                f"data/prepare_mitbih.py first to build them from raw "
                f"MIT-BIH WFDB records."
            )
        self.X = np.load(x_path).astype(np.float32)  # (N, T)
        self.y = np.load(y_path).astype(np.int64)     # (N,)
        assert len(self.X) == len(self.y), "X/y length mismatch"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sig = self.X[idx]
        sig = normalize_signal(sig)
        return torch.from_numpy(sig).unsqueeze(0), self.y[idx]  # (1, T)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                         help="Directory with X.npy, y.npy from prepare_mitbih.py")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--patient-independent-split", action="store_true",
                         help="Strongly recommended: ensures split.py "
                              "was used to separate patients between "
                              "train/val, avoiding the same-patient "
                              "leakage the manuscript claims to avoid.")
    parser.add_argument("--out", default="ta_cnn_checkpoint.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = BeatWindowDataset(args.data_dir)
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    num_classes = int(dataset.y.max()) + 1
    model = TACNN(in_channels=1, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, f1, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_acc={acc:.4f} | val_f1_macro={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved new best checkpoint to {args.out}")

    print(f"\nTraining complete. Best val F1 (macro): {best_f1:.4f}")
    print("This number is real output from this run on your data -- "
          "it is NOT guaranteed to match any figure previously written "
          "in the manuscript. Report whatever this script actually "
          "prints.")


if __name__ == "__main__":
    main()
