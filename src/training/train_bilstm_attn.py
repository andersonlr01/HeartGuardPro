"""
Training loop for BiLSTM-Attn on HRV feature sequences derived from
MIMIC-III waveform records (internal cohort) with PhysioNet 2019 held
out for external validation (run
src/evaluation/external_validation.py separately after this).

REQUIRES REAL DATA: run data/prepare_mimic3.py first to build HRV
feature-sequence tensors from raw MIMIC-III waveform records (requires
credentialed PhysioNet access -- see data/README.md).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.bilstm_attn import BiLSTMAttn


class HRVSequenceDataset(Dataset):
    """Loads (sequence_of_HRV_vectors, binary_instability_label) pairs.

    Expects X.npy of shape (N, T, F) -- N windows, T timesteps,
    F = number of HRV features (paper uses RMSSD, SDNN, SD1, SD2,
    SD2/SD1, LF/HF, SampEn -> F=7) -- and y.npy of shape (N,) with
    0/1 labels for "no instability" / "instability within horizon".
    """

    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        x_path, y_path = data_dir / "X.npy", data_dir / "y.npy"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Expected {x_path} and {y_path}. Run "
                f"data/prepare_mimic3.py first."
            )
        self.X = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), self.y[idx]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        preds.extend(out.argmax(dim=1).cpu().numpy())
        labels.extend(yb.numpy())
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    return f1, precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True,
                         help="Patient-disjoint validation split (see "
                              "data/patient_split.py)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out", default="bilstm_attn_checkpoint.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = HRVSequenceDataset(args.train_dir)
    val_ds = HRVSequenceDataset(args.val_dir)
    feature_dim = train_ds.X.shape[-1]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = BiLSTMAttn(input_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # class weighting for imbalance (instability events are rare)
    labels = train_ds.y
    pos_weight = (labels == 0).sum() / max((labels == 1).sum(), 1)
    class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        f1, prec, rec = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_f1={f1:.4f} | val_precision={prec:.4f} | val_recall={rec:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.out)

    print(f"\nBest internal validation F1: {best_f1:.4f}")
    print("Run src/evaluation/external_validation.py with this checkpoint "
          "against PhysioNet 2019 data for the external generalization "
          "figure -- do not reuse the manuscript's previous number.")


if __name__ == "__main__":
    main()
