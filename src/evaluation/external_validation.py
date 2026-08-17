"""
External validation: evaluate a BiLSTM-Attn checkpoint trained on
MIMIC-III against a held-out PhysioNet 2019 cohort, reporting the
generalization gap (manuscript Section V.D).

REQUIRES REAL DATA: run data/prepare_physionet2019.py first.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.bilstm_attn import BiLSTMAttn
from src.training.train_bilstm_attn import HRVSequenceDataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--internal-val-dir", required=True,
                         help="MIMIC-III internal validation split")
    parser.add_argument("--external-dir", required=True,
                         help="PhysioNet 2019 external cohort, from "
                              "data/prepare_physionet2019.py")
    parser.add_argument("--feature-dim", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttn(input_dim=args.feature_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    def eval_f1(data_dir):
        ds = HRVSequenceDataset(data_dir)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                out = model(xb.to(device))
                preds.extend(out.argmax(dim=1).cpu().numpy())
                labels.extend(yb.numpy())
        return f1_score(labels, preds)

    internal_f1 = eval_f1(args.internal_val_dir)
    external_f1 = eval_f1(args.external_dir)
    gap_pp = (internal_f1 - external_f1) * 100

    print(f"Internal (MIMIC-III) F1:        {internal_f1:.4f}")
    print(f"External (PhysioNet 2019) F1:   {external_f1:.4f}")
    print(f"Generalization gap:             {gap_pp:.1f} percentage points")
    print("\nThese are the real numbers for THIS checkpoint on THIS data. "
          "To claim a comparative generalization-gap advantage over "
          "baselines (as previously drafted), you must run this same "
          "script against each trained baseline checkpoint and compare "
          "honestly -- see src/models/baselines.py.")


if __name__ == "__main__":
    main()
