"""
Baseline architectures for fair comparison against BiLSTM-Attn, as
listed in the manuscript's comparison table: CNN-only, standard
BiLSTM (no attention), Transformer-only, and InceptionTime.

Train each of these with src/training/train_bilstm_attn.py by
swapping the model class (or write an analogous train_baseline.py),
on the IDENTICAL data split used for BiLSTM-Attn, then feed the paired
predictions into src/evaluation/statistical_tests.mcnemar_test for a
genuine significance comparison.
"""
import torch
import torch.nn as nn


class CNNOnly(nn.Module):
    """1D-CNN baseline with no recurrence or attention."""

    def __init__(self, input_dim: int, num_classes: int = 2, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):  # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.net(x).squeeze(-1)
        return self.fc(x)


class StandardBiLSTM(nn.Module):
    """BiLSTM without the self-attention block -- isolates the
    contribution of attention when compared against BiLSTMAttn."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TransformerOnly(nn.Module):
    """Transformer encoder baseline (no recurrence)."""

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :])


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, 1, padding=0)
        self.branch3 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.branch5 = nn.Conv1d(in_channels, out_channels, 5, padding=2)
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, 1),
        )
        self.bn = nn.BatchNorm1d(out_channels * 4)

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch3(x),
                          self.branch5(x), self.branch_pool(x)], dim=1)
        return torch.relu(self.bn(out))


class InceptionTime(nn.Module):
    """Simplified InceptionTime (Ismail Fawaz et al., ref [14]) for
    fair-comparison purposes -- a compact version, not a full
    reimplementation of the original repo. For a rigorous comparison
    prefer the authors' reference implementation."""

    def __init__(self, input_dim: int, num_classes: int = 2, out_channels: int = 32):
        super().__init__()
        self.m1 = InceptionModule(input_dim, out_channels)
        self.m2 = InceptionModule(out_channels * 4, out_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels * 4, num_classes)

    def forward(self, x):  # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.m1(x)
        x = self.m2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
