"""Shared model + constants for the Gulf Watch CNN+LSTM predictor.

Both `gulf_watch_notebook.py` (training) and `serving_wrapper.py` (inference
endpoint) import from here so the two stay in lock-step. If you change a
hyper-parameter or layer here, retrain — don't fork.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Geographic constants (Gulf of Mexico)
# --------------------------------------------------------------------------- #
LAT_MIN, LAT_MAX = 22.0, 31.0
LON_MIN, LON_MAX = -97.0, -80.0

# Loop Current zone — primary ML target
LC_LAT_MIN, LC_LAT_MAX = 23.0, 27.0
LC_LON_MIN, LC_LON_MAX = -87.0, -83.0

# Western Gulf reference zone (for separation-gradient label)
WG_LAT_MIN, WG_LAT_MAX = 23.0, 27.0
WG_LON_MIN, WG_LON_MAX = -95.0, -90.0

# --------------------------------------------------------------------------- #
# Science constants
# --------------------------------------------------------------------------- #
RI_THRESHOLD = 0.17        # m SSH anomaly — Mainelli et al. 2008 RI proxy
GRADIENT_THRESHOLD = 0.10  # m — LCE separation gradient (LC zone − western Gulf)
WARMING_OFFSET = 0.05      # m SSH per +2°C SST (thermal expansion proxy)

# --------------------------------------------------------------------------- #
# Model architecture
# --------------------------------------------------------------------------- #
SEQ_LEN = 30
HIDDEN_SIZE = 64
DROPOUT = 0.2
CNN_FEAT_DIM = 128
CNN_IN_CHANNELS = 3   # raw SSH anomaly, |∇SSH|, 7-day Δ


class SSHEncoder(nn.Module):
    """Encodes a multi-channel SSH field into a 128-dim feature vector.

    Channels (physics-informed, all derived from the raw SSH anomaly):
      0 — SSH anomaly (Loop-Current warm bulge)
      1 — |∇SSH| gradient magnitude (fronts + eddy edges)
      2 — 7-day SSH-anomaly change (dynamical spin-up / decay)
    """

    def __init__(self, h: int, w: int, in_ch: int = CNN_IN_CHANNELS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, in_ch, h, w)).view(1, -1).shape[1]
        self.fc = nn.Linear(flat, CNN_FEAT_DIM)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))


class LCEPredictor(nn.Module):
    """30-day LSTM → (p7, p30) sigmoid heads."""

    def __init__(self, input_size: int = CNN_FEAT_DIM,
                 hidden_size: int = HIDDEN_SIZE,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc7 = nn.Linear(hidden_size, 1)
        self.fc30 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = self.drop(out[:, -1, :])
        return (
            torch.sigmoid(self.fc7(last)).squeeze(-1),
            torch.sigmoid(self.fc30(last)).squeeze(-1),
        )


# --------------------------------------------------------------------------- #
# Channel-builder used at training AND inference time. The frozen random-init
# encoder requires identical preprocessing across both, otherwise its random
# projection means something different on inference inputs.
# --------------------------------------------------------------------------- #

def build_three_channels(
    raw: np.ndarray,
    start: int,
    end: int,
    std_ch: np.ndarray,
    lag: int,
) -> np.ndarray:
    """Build (batch, 3, H, W) float32 channels for frames `raw[start:end]`.

    Parameters
    ----------
    raw : (T, H, W) float32 nan-filled SSH anomaly tensor.
    start, end : frame indices (end exclusive).
    std_ch : (3,) per-channel std saved at training time.
    lag : integer frame lag for the Δ channel (e.g. round(7 / subsample_days)).
    """
    block = raw[start:end]
    gy, gx = np.gradient(block, axis=(1, 2))
    gmag = np.sqrt(gy ** 2 + gx ** 2).astype(np.float32)
    tr = np.zeros_like(block)
    for k, i in enumerate(range(start, end)):
        if i - lag >= 0:
            tr[k] = raw[i] - raw[i - lag]
    out = np.stack([block, gmag, tr], axis=1)
    out[:, 0] /= std_ch[0]
    out[:, 1] /= std_ch[1]
    out[:, 2] /= std_ch[2]
    return out


# --------------------------------------------------------------------------- #
# Risk-heatmap math (deterministic transform of an SSH anomaly frame).
# Used by the notebook's risk_heatmap() AND by serving_wrapper for ri_probability.
# --------------------------------------------------------------------------- #

def risk_field_from_anomaly(
    field: np.ndarray,
    sigma: float = 1.2,
) -> np.ndarray:
    """Map an SSH anomaly field (m) → smoothed risk field in [0, 1].

    Mirrors notebook Cell 9 exactly so the live endpoint and the training
    plots agree pixel-for-pixel.
    """
    from scipy.ndimage import gaussian_filter

    field = np.nan_to_num(field, nan=0.0)
    ssh_max = max(float(np.abs(field).max()), 1e-6)
    risk = np.clip(field / ssh_max, 0.0, 1.0)
    return gaussian_filter(risk, sigma=sigma)


# --------------------------------------------------------------------------- #
# Highest-risk-zone label table. Keyed by (lat, lon) so the wrapper can match
# the argmax of the risk grid to a human-readable region.
# --------------------------------------------------------------------------- #

_REGION_TABLE = [
    # (name, lat_lo, lat_hi, lon_lo, lon_hi)
    ("Loop Current core, ~120mi west of Tampa",     24.5, 27.0, -87.0, -83.0),
    ("Eastern Gulf, ~150mi south of Tampa",         23.0, 26.5, -85.0, -82.0),
    ("Northern Gulf, off the Mississippi Delta",    27.5, 30.0, -91.0, -87.0),
    ("Western Gulf, off the Texas coast",           24.0, 29.0, -97.0, -93.0),
    ("Bay of Campeche",                             19.0, 23.0, -95.0, -90.0),
    ("Florida Straits approach",                    23.5, 25.5, -83.0, -80.0),
]


def label_region(lat: float, lon: float) -> str:
    """Map a (lat, lon) point to a human-readable Gulf region label.

    Falls back to 'Central Gulf basin' if no zone matches.
    """
    # Normalize 0–360° longitudes back to ±180° for the lookup
    if lon > 180:
        lon = lon - 360
    for name, lat_lo, lat_hi, lon_lo, lon_hi in _REGION_TABLE:
        if lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi:
            return name
    return "Central Gulf basin"
