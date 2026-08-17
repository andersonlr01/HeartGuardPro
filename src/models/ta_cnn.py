"""
TA-CNN: Temporal Attention-enhanced 1D-CNN for edge arrhythmia
classification (manuscript Section III, edge model).

Designed to be small enough to quantize to INT8 and deploy on an
ESP32-S3 (target: well under a ~200 KB flash budget per the paper's
related-work discussion). Actual on-device latency (paper claims
14.2 ms) can only be measured on real hardware after TFLite Micro
conversion -- see src/edge/convert_tflite.py. Any latency number
quoted before that conversion-and-benchmark step is a projection, not
a measurement.
"""
import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Channel-wise temporal attention over a 1D feature map.

    Learns a soft weighting over the time axis for each channel,
    letting the network emphasize the QRS-like transient regions of a
    beat window.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels // 4 if channels >= 4 else 1,
                      kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4 if channels >= 4 else 1, channels,
                      kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):  # x: (B, C, T)
        weights = self.attn(x)  # (B, C, T), in (0,1)
        return x * weights


class TACNN(nn.Module):
    """Lightweight 1D-CNN with temporal attention for beat-level
    arrhythmia classification (e.g. MIT-BIH AAMI 5-class scheme:
    N, S, V, F, Q).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 5,
                 base_channels: int = 16):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.attn1 = TemporalAttention(c1)

        self.block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.attn2 = TemporalAttention(c2)

        self.block3 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
        )
        self.attn3 = TemporalAttention(c3)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x):  # x: (B, in_channels, T)
        x = self.attn1(self.block1(x))
        x = self.attn2(self.block2(x))
        x = self.attn3(self.block3(x))
        x = self.global_pool(x).squeeze(-1)  # (B, c3)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_fp32_size_kb(self) -> float:
        """Rough size estimate BEFORE INT8 quantization (params * 4
        bytes). Real flash footprint must be measured after TFLite
        Micro conversion (see src/edge/convert_tflite.py)."""
        return self.count_parameters() * 4 / 1024.0


if __name__ == "__main__":
    model = TACNN(in_channels=1, num_classes=5, base_channels=16)
    dummy = torch.randn(8, 1, 250)  # batch=8, 1-lead, 250 samples/beat
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Trainable parameters: {model.count_parameters():,}")
    print(f"Estimated FP32 size: {model.estimate_fp32_size_kb():.1f} KB "
          f"(will shrink ~4x after INT8 quantization)")
