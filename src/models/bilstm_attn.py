"""
BiLSTM-Attn: Bidirectional LSTM with multi-head self-attention for
long-horizon cardiovascular instability forecasting (manuscript
Section III, cloud model).

Consumes a sequence of HRV feature vectors (see
src/preprocessing/hrv_features.py) sampled over a lookback window and
predicts instability risk at a configurable forecast horizon.
"""
import torch
import torch.nn as nn


class BiLSTMAttn(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.3, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2  # bidirectional

        self.self_attn = nn.MultiheadAttention(
            embed_dim=lstm_out_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(lstm_out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, num_classes),
        )

    def forward(self, x, return_attn: bool = False):
        # x: (B, T, input_dim) -- sequence of HRV feature vectors
        lstm_out, _ = self.lstm(x)  # (B, T, 2*hidden_dim)

        attn_out, attn_weights = self.self_attn(
            lstm_out, lstm_out, lstm_out, need_weights=True,
            average_attn_weights=True,
        )
        fused = self.layer_norm(lstm_out + attn_out)  # residual + norm

        # Pool over time: use the last time step (most recent
        # observation before the forecast horizon)
        pooled = fused[:, -1, :]
        logits = self.classifier(pooled)

        if return_attn:
            return logits, attn_weights
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = BiLSTMAttn(input_dim=7, hidden_dim=64, num_layers=2,
                        num_heads=4, num_classes=2)
    dummy = torch.randn(8, 30, 7)  # batch=8, 30 timesteps, 7 HRV features
    logits = model(dummy)
    print(f"Output shape: {logits.shape}")
    print(f"Trainable parameters: {model.count_parameters():,}")
