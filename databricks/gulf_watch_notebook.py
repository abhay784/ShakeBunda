# =============================================================================
# GULF WATCH — Databricks Notebook
# CNN + LSTM LCE Separation Predictor + RI Risk Heatmap
# Scripps / ECCO Gulf of Mexico 40-year SSH Dataset
# DataHacks 2026 | Climate + AI/ML Track
# =============================================================================
# Copy each `# CELL N` block into a separate Databricks cell and run top-to-bottom.
# Each `# ✅ DEMO CHECKPOINT` line marks a safe stopping point.
# =============================================================================


# CELL 1 — IMPORTS & CONFIG
# =============================================================================

%pip install xarray netCDF4 torch torchvision matplotlib cartopy scikit-learn --quiet

import os, warnings, json
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score, f1_score, roc_curve

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# --------------------------------------------------------------------------- #
# ⚠️ SWAP THIS: update path to match where .nc file was uploaded in DBFS
NC_PATH = "/dbfs/FileStore/gulf_watch/run2_clim_v2_ssh.nc"
# Alt (if mounted):  "/mnt/gulf-watch/run2_clim_v2_ssh.nc"
# --------------------------------------------------------------------------- #

# Gulf of Mexico bounding box
LAT_MIN, LAT_MAX = 22.0, 31.0
LON_MIN, LON_MAX = -97.0, -80.0

# Loop Current zone — primary ML target
LC_LAT_MIN, LC_LAT_MAX = 23.0, 27.0
LC_LON_MIN, LC_LON_MAX = -87.0, -83.0

# Western Gulf reference zone (for separation-gradient label)
WG_LAT_MIN, WG_LAT_MAX = 23.0, 27.0
WG_LON_MIN, WG_LON_MAX = -95.0, -90.0

# 🔬 SCIENCE NOTE: 0.17 m SSH anomaly threshold = established RI proxy
# (Shay et al. 2000, Mainelli et al. 2008)
RI_THRESHOLD = 0.17
GRADIENT_THRESHOLD = 0.10   # LCE separation: LC minus western Gulf

# +2°C warming proxy — thermal expansion ~ +0.05 m SSH
WARMING_OFFSET = 0.05

# Model config
SEQ_LEN = 30
HIDDEN_SIZE = 64
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

USE_CNN = True   # ⚡ FALLBACK: set False if CNN feature extraction is too slow

PLOT_DIR = "/tmp"
os.makedirs(PLOT_DIR, exist_ok=True)

print("✅ Imports + config OK")
print(f"   PyTorch={torch.__version__}  xarray={xr.__version__}")
print(f"   USE_CNN={USE_CNN}  plots→{PLOT_DIR}")

# ✅ DEMO CHECKPOINT: safe stopping point if time runs out


# CELL 2 — DATA LOADING & INSPECTION
# =============================================================================

# Chunked load — Community Edition RAM can't hold full 40-year array
ds = xr.open_dataset(NC_PATH, chunks={"time": 365})

print("── Dataset ──────────────────────────────────────────")
print(ds)

# ⚠️ SWAP THIS — variable name auto-detect
SSH_CANDIDATES  = ["ssh", "eta", "zos", "sea_surface_height", "SSH", "ETA", "adt"]
LAT_CANDIDATES  = ["lat", "latitude", "y", "YC"]
LON_CANDIDATES  = ["lon", "longitude", "x", "XC"]
TIME_CANDIDATES = ["time", "TIME", "t"]

def _first_present(candidates, pool):
    return next((n for n in candidates if n in pool), None)

ssh_var  = _first_present(SSH_CANDIDATES, ds.data_vars)
lat_dim  = _first_present(LAT_CANDIDATES, list(ds.coords) + list(ds.dims))
lon_dim  = _first_present(LON_CANDIDATES, list(ds.coords) + list(ds.dims))
time_dim = _first_present(TIME_CANDIDATES, list(ds.coords) + list(ds.dims))

if ssh_var is None:
    print("Available variables:", list(ds.data_vars))
    raise ValueError("SSH variable not found — update SSH_CANDIDATES above")
if not all([lat_dim, lon_dim, time_dim]):
    raise ValueError(f"Missing coord(s): lat={lat_dim} lon={lon_dim} time={time_dim}")

print(f"\n✅ Detected: ssh='{ssh_var}' lat='{lat_dim}' lon='{lon_dim}' time='{time_dim}'")
print(f"   Time range : {ds[time_dim].values[0]} → {ds[time_dim].values[-1]}")
print(f"   Grid shape : lat={len(ds[lat_dim])}  lon={len(ds[lon_dim])}")

# Longitude convention
if ds[lon_dim].values.max() > 180:
    lon_min_q, lon_max_q = LON_MIN + 360, LON_MAX + 360
    print(f"   Longitude : 0–360° convention detected → slicing [{lon_min_q}, {lon_max_q}]")
else:
    lon_min_q, lon_max_q = LON_MIN, LON_MAX
    print(f"   Longitude : ±180° convention → slicing [{lon_min_q}, {lon_max_q}]")

# Crop to Gulf of Mexico
ssh_gulf = ds[ssh_var].sel(
    {lat_dim: slice(LAT_MIN, LAT_MAX),
     lon_dim: slice(lon_min_q, lon_max_q)}
)
print(f"\n✅ Gulf crop shape: {ssh_gulf.shape}")

print("Loading Gulf crop into memory …")
ssh_gulf = ssh_gulf.load()
print(f"✅ In-memory  |  min={float(ssh_gulf.min()):.3f}  max={float(ssh_gulf.max()):.3f} m")

# ✅ DEMO CHECKPOINT: safe stopping point if time runs out


# CELL 3 — ANCHOR MOMENT VERIFICATION
# =============================================================================

ANCHORS = [
    ("Katrina", "2005-08-28", -87.5, 25.5, "anchor_katrina"),
    ("Harvey",  "2017-08-24", -94.0, 25.5, "anchor_harvey"),
    ("Ida",     "2021-08-28", -90.5, 26.8, "anchor_ida"),
]

def _nearest_date(da, target):
    t = np.datetime64(target)
    times = da[time_dim].values
    i = int(np.argmin(np.abs(times - t)))
    return da.isel({time_dim: i}), str(times[i])[:10]

for name, date_str, slon, slat, fname in ANCHORS:
    try:
        frame, actual = _nearest_date(ssh_gulf, date_str)
        lons_arr = frame[lon_dim].values
        lats_arr = frame[lat_dim].values

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.pcolormesh(lons_arr, lats_arr, frame.values,
                           cmap="RdBu_r", vmin=-0.4, vmax=0.4, shading="auto")
        ax.contour(lons_arr, lats_arr, frame.values,
                   levels=[RI_THRESHOLD], colors="red", linewidths=2, linestyles="--")
        ax.plot(slon if lons_arr.max() <= 180 else slon + 360, slat,
                "w*", markersize=20, markeredgecolor="black", markeredgewidth=1,
                label=f"{name} position")

        # 🎯 KATRINA CHECK (and Harvey/Ida equivalents)
        lat_sl = slice(slat - 2, slat + 2)
        lon_probe = slon if lons_arr.max() <= 180 else slon + 360
        lon_sl = slice(lon_probe - 2, lon_probe + 2)
        local_max = float(frame.sel({lat_dim: lat_sl, lon_dim: lon_sl}).max())
        verdict = "✅ warm eddy" if local_max > RI_THRESHOLD else "⚠️ weak signal"

        ax.set_title(f"{name}  —  {actual}\nLocal max SSH={local_max:.2f}m  {verdict}",
                     fontsize=11)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        plt.colorbar(im, ax=ax, label="SSH (m)")
        ax.legend(loc="lower right")
        plt.tight_layout()

        out = f"{PLOT_DIR}/{fname}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"✅ {name} ({actual})  local_max={local_max:.3f}m  →  {out}")
    except Exception as e:
        print(f"⚠️ {name} {date_str}: {e}")

# ✅ DEMO CHECKPOINT: safe stopping point if time runs out


# CELL 4 — FEATURE ENGINEERING
# =============================================================================

print("Computing long-term SSH mean per grid point …")
ssh_mean = ssh_gulf.mean(dim=time_dim)
ssh_anomaly = ssh_gulf - ssh_mean
print(f"✅ ssh_anomaly shape={ssh_anomaly.shape}  "
      f"range=[{float(ssh_anomaly.min()):.3f}, {float(ssh_anomaly.max()):.3f}] m")

# 🔬 SCIENCE NOTE: RI risk flag is per-pixel threshold crossing
ri_risk = (ssh_anomaly > RI_THRESHOLD).astype(float)
print(f"✅ ri_risk field  mean fraction above threshold = "
      f"{float(ri_risk.mean()):.3%}")

# Adjust LC / WG longitude slicing for 0–360 convention
def _lon_slice(a, b):
    if ssh_anomaly[lon_dim].values.max() > 180:
        return slice(a + 360, b + 360)
    return slice(a, b)

lc_ts = ssh_anomaly.sel(
    {lat_dim: slice(LC_LAT_MIN, LC_LAT_MAX),
     lon_dim: _lon_slice(LC_LON_MIN, LC_LON_MAX)}
).mean(dim=[lat_dim, lon_dim])

wg_ts = ssh_anomaly.sel(
    {lat_dim: slice(WG_LAT_MIN, WG_LAT_MAX),
     lon_dim: _lon_slice(WG_LON_MIN, WG_LON_MAX)}
).mean(dim=[lat_dim, lon_dim])

lc_series = pd.Series(lc_ts.values,
                      index=pd.to_datetime(lc_ts[time_dim].values),
                      name="lc_ssh_anom")
wg_series = pd.Series(wg_ts.values,
                      index=lc_series.index, name="wg_ssh_anom")

rolling_7  = lc_series.rolling(7,  min_periods=1).mean()
rolling_30 = lc_series.rolling(30, min_periods=7).mean()

print(f"✅ LC time series: n={len(lc_series)}  "
      f"[{lc_series.index[0].date()} … {lc_series.index[-1].date()}]")

# ✅ DEMO CHECKPOINT: safe stopping point if time runs out


# CELL 5 — LCE SEPARATION LABELS (self-supervised)
# =============================================================================

# 🔬 SCIENCE NOTE: LCE separation = warm bulge in LC zone AND a gradient break
# between LC and western Gulf (eddy has detached and drifted west)
gradient = lc_series - wg_series
raw_label = ((lc_series > RI_THRESHOLD) & (gradient > GRADIENT_THRESHOLD)).astype(float)
smooth = raw_label.rolling(5, center=True).mean().fillna(0)
sep_label = (smooth > 0.5).astype(int)

print(f"✅ Separation labels  positives={int(sep_label.sum())} / {len(sep_label)}  "
      f"({100*sep_label.mean():.2f}%)")

# Visual sanity — label timeline with storm dates
fig, ax = plt.subplots(figsize=(16, 4))
ax.fill_between(lc_series.index, lc_series.values, 0,
                where=lc_series.values > 0, color="red", alpha=0.3, label="LC SSH anomaly (+)")
ax.fill_between(lc_series.index, lc_series.values, 0,
                where=lc_series.values <= 0, color="blue", alpha=0.3, label="LC SSH anomaly (−)")
ax.plot(rolling_30.index, rolling_30.values, "k--", lw=1, label="30-day rolling")
ax.axhline(RI_THRESHOLD, color="orange", ls=":", label=f"RI threshold ({RI_THRESHOLD}m)")

# Shade separation windows
ax.fill_between(sep_label.index, 0, 1,
                where=sep_label.values > 0, transform=ax.get_xaxis_transform(),
                color="purple", alpha=0.15, label="LCE separation label")
for name, d, *_ in ANCHORS:
    ax.axvline(pd.Timestamp(d), color="black", lw=1)
    ax.text(pd.Timestamp(d), ax.get_ylim()[1]*0.85, name,
            rotation=90, fontsize=7)
ax.set_title("Loop Current SSH Anomaly + Self-supervised LCE Separation Labels")
ax.set_xlabel("Date"); ax.set_ylabel("SSH Anomaly (m)")
ax.legend(ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/separation_labels.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Saved → {PLOT_DIR}/separation_labels.png")

# ✅ DEMO CHECKPOINT: safe stopping point if time runs out


# CELL 6 — CNN ENCODER
# =============================================================================

class SSHEncoder(nn.Module):
    """Encodes a multi-channel SSH field into a 128-dim feature vector.

    Channels (physics-informed, all derived from the raw SSH anomaly):
      0 — SSH anomaly (Loop-Current warm bulge)
      1 — |∇SSH| gradient magnitude (fronts + eddy edges; RI often initiates here)
      2 — 7-day SSH-anomaly change (dynamical spin-up / decay)
    """
    def __init__(self, h, w, in_ch=3):
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
        self.fc = nn.Linear(flat, 128)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Memory-safe: compute the 3 channels per-batch and stream into the encoder,
# so we never materialise the full (T, 3, H, W) tensor.
raw = np.nan_to_num(ssh_anomaly.values, nan=0.0).astype(np.float32)
T_, H, W = raw.shape
print(f"✅ Grid size: H={H}  W={W}  T={T_}  (device={device})")

# Lag for Δ-channel: data is subsampled 3-day (Cell 2) → 2 steps ≈ 6 days
_lag = max(1, int(round(7.0 / 3.0)))

# Estimate per-channel std on a random subset (avoids full-array alloc)
_rng = np.random.default_rng(0)
_sample_idx = _rng.choice(T_, size=min(256, T_), replace=False)
_samp_raw = raw[_sample_idx]
_gy, _gx = np.gradient(_samp_raw, axis=(1, 2))
_samp_grad = np.sqrt(_gy ** 2 + _gx ** 2)
_pairs = [(i, i - _lag) for i in _sample_idx if i - _lag >= 0]
_samp_trend = (raw[[a for a, _ in _pairs]] -
               raw[[b for _, b in _pairs]]) if _pairs else np.zeros_like(_samp_raw)
std_ch = np.array([
    _samp_raw.std()  + 1e-6,
    _samp_grad.std() + 1e-6,
    _samp_trend.std() + 1e-6,
], dtype=np.float32)
del _samp_raw, _samp_grad, _samp_trend, _gy, _gx
print(f"   channel stds (raw, |∇|, Δ{_lag*3}d): "
      f"{std_ch[0]:.4f}  {std_ch[1]:.4f}  {std_ch[2]:.4f}")

def _build_batch_channels(start, end):
    """Return (batch, 3, H, W) float32 for frames [start, end)."""
    block = raw[start:end]
    gy, gx = np.gradient(block, axis=(1, 2))
    gmag = np.sqrt(gy ** 2 + gx ** 2).astype(np.float32)
    tr = np.zeros_like(block)
    for k, i in enumerate(range(start, end)):
        if i - _lag >= 0:
            tr[k] = raw[i] - raw[i - _lag]
    out = np.stack([block, gmag, tr], axis=1)
    out[:, 0] /= std_ch[0]
    out[:, 1] /= std_ch[1]
    out[:, 2] /= std_ch[2]
    return out

if USE_CNN:
    print("Encoding multi-channel SSH fields with CNN (streamed) …")
    encoder = SSHEncoder(H, W, in_ch=3).to(device).eval()
    feats = np.zeros((T_, 128), dtype=np.float32)
    step = 128
    with torch.no_grad():
        for i in range(0, T_, step):
            batch_np = _build_batch_channels(i, min(i + step, T_))
            batch = torch.from_numpy(batch_np).to(device)
            feats[i:i + batch_np.shape[0]] = encoder(batch).cpu().numpy()
            del batch, batch_np
    print(f"✅ CNN features: {feats.shape}")
else:
    # ⚡ FALLBACK: scalar 3-feature time series
    feats = np.stack([
        lc_series.values,
        rolling_7.fillna(0).values,
        rolling_30.fillna(0).values,
    ], axis=1).astype(np.float32)
    print(f"✅ Fallback scalar features: {feats.shape}")

INPUT_SIZE = feats.shape[1]

# ✅ DEMO CHECKPOINT: safe stopping point if time runs out


# CELL 7 — LSTM PREDICTOR (trained on IBTrACS RI labels)
# =============================================================================
# Swaps the self-supervised "LCE separation" proxy for direct supervision on
# NOAA IBTrACS rapid-intensification (RI) onsets within the Gulf. Handles
# class imbalance via WeightedRandomSampler. Early-stops on val ROC-AUC and
# restores best weights.

from torch.utils.data import WeightedRandomSampler

IBTRACS_PATH     = "/Volumes/workspace/default/gulf_watch/ibtracs.NA.list.v04r01.csv"
RI_WIND_DELTA_KT = 30.0
RI_WINDOW_HOURS  = 24
GULF_BBOX        = {"lat": (18.0, 32.0), "lon": (-98.0, -78.0)}
EARLY_STOP_PATIENCE = 5

# ------- Load IBTrACS and build Gulf-only RI-onset day vector -------
assert os.path.exists(IBTRACS_PATH), f"Missing {IBTRACS_PATH}"
_ib = pd.read_csv(IBTRACS_PATH, skiprows=[1], low_memory=False,
                  na_values=[" ", "", "NA"])
_ib["ISO_TIME"] = pd.to_datetime(_ib["ISO_TIME"], errors="coerce")
for _c in ("LAT", "LON", "USA_WIND", "WMO_WIND"):
    if _c in _ib.columns:
        _ib[_c] = pd.to_numeric(_ib[_c], errors="coerce")
_ib["WIND_KT"] = _ib["USA_WIND"].fillna(_ib["WMO_WIND"])
_ib = _ib[_ib["SEASON"] >= 1982].dropna(
    subset=["WIND_KT", "ISO_TIME", "LAT", "LON"])
_gulf = _ib[_ib["LAT"].between(*GULF_BBOX["lat"]) &
            _ib["LON"].between(*GULF_BBOX["lon"])].copy()

def _flag_ri(g):
    g = g.sort_values("ISO_TIME").reset_index(drop=True)
    t = g["ISO_TIME"].values
    w = g["WIND_KT"].values
    flag = np.zeros(len(g), dtype=bool)
    for i in range(len(g)):
        dt_h = (t - t[i]).astype("timedelta64[h]").astype(float)
        later = np.where(dt_h >= RI_WINDOW_HOURS)[0]
        if len(later) == 0:
            break
        j = later[0]
        if w[j] - w[i] >= RI_WIND_DELTA_KT:
            flag[i] = True
    return g.assign(RI_FLAG=flag)

_gulf = _gulf.groupby("SID", group_keys=False).apply(_flag_ri)
_ri_onset_days = pd.DatetimeIndex(
    pd.to_datetime(_gulf.loc[_gulf["RI_FLAG"], "ISO_TIME"]).dt.floor("D").unique()
).sort_values()
print(f"✅ IBTrACS Gulf RI onset days since 1982: {len(_ri_onset_days)}")

# Align onset days to the SSH time axis (subsampled by 3 days in Cell 2)
_onset_set = set(_ri_onset_days.values.astype("datetime64[D]"))
_day_dates = lc_series.index.values.astype("datetime64[D]")

# For each model day t, label True if ANY day in [t, t + window] is an onset.
# This accommodates the 3-day subsample so onsets on off-grid days still score.
def _window_label(window_days):
    y = np.zeros(len(_day_dates), dtype=np.float32)
    for k, d0 in enumerate(_day_dates):
        for offset in range(1, window_days + 1):
            if (d0 + np.timedelta64(offset, "D")) in _onset_set:
                y[k] = 1.0
                break
    return y

y7_full  = _window_label(7)
y30_full = _window_label(30)
print(f"   Label base rates  y7={y7_full.mean():.2%}  y30={y30_full.mean():.2%}")

class LCEPredictor(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc7  = nn.Linear(hidden_size, 1)
        self.fc30 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = self.drop(out[:, -1, :])
        return (torch.sigmoid(self.fc7(last)).squeeze(-1),
                torch.sigmoid(self.fc30(last)).squeeze(-1))

# Build (X, y7, y30) sequences
T = feats.shape[0]
X_list, y7_list, y30_list = [], [], []
for i in range(SEQ_LEN, T - 30):
    X_list.append(feats[i - SEQ_LEN : i])
    y7_list.append(y7_full[i])
    y30_list.append(y30_full[i])

X_all   = np.asarray(X_list,   dtype=np.float32)
y7_all  = np.asarray(y7_list,  dtype=np.float32)
y30_all = np.asarray(y30_list, dtype=np.float32)
print(f"✅ Sequences: X={X_all.shape}  y7 pos={y7_all.mean():.2%}  "
      f"y30 pos={y30_all.mean():.2%}")

# Chronological 38yr / remainder split
split = int(len(X_all) * (38.0 / 42.0))
X_tr, X_val = X_all[:split], X_all[split:]
y7_tr, y7_val = y7_all[:split], y7_all[split:]
y30_tr, y30_val = y30_all[:split], y30_all[split:]
print(f"   Train={len(X_tr)} (pos7={y7_tr.mean():.2%})  "
      f"Val={len(X_val)} (pos7={y7_val.mean():.2%})")
assert y7_tr.sum() > 0 and y7_val.sum() > 0, "Split has zero positives — retune window"

# Weighted sampler: oversample positives to ~50/50 in each batch
_p = max(float(y7_tr.mean()), 1e-6)
sample_w = np.where(y7_tr > 0, 1.0 / _p, 1.0 / (1.0 - _p)).astype(np.float64)
sampler = WeightedRandomSampler(
    torch.tensor(sample_w, dtype=torch.double),
    num_samples=len(sample_w), replacement=True,
)

tr_loader = DataLoader(
    TensorDataset(torch.tensor(X_tr), torch.tensor(y7_tr), torch.tensor(y30_tr)),
    batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val), torch.tensor(y7_val), torch.tensor(y30_val)),
    batch_size=BATCH_SIZE, shuffle=False)

model = LCEPredictor(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                     dropout=DROPOUT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss()

train_losses, val_losses = [], []
best_auc, best_state, stale = -1.0, None, 0
print(f"Training for up to {EPOCHS} epochs (early stop on val AUC) …")
for epoch in range(1, EPOCHS + 1):
    model.train(); running = 0.0
    for xb, y7b, y30b in tr_loader:
        xb = xb.to(device); y7b = y7b.to(device); y30b = y30b.to(device)
        p7, p30 = model(xb)
        loss = loss_fn(p7, y7b) + loss_fn(p30, y30b)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.item() * len(xb)

    model.eval(); vrun = 0.0; vp7, vy7 = [], []
    with torch.no_grad():
        for xb, y7b, y30b in val_loader:
            xb = xb.to(device); y7b = y7b.to(device); y30b = y30b.to(device)
            p7, p30 = model(xb)
            vrun += (loss_fn(p7, y7b) + loss_fn(p30, y30b)).item() * len(xb)
            vp7.append(p7.cpu().numpy()); vy7.append(y7b.cpu().numpy())
    vp7 = np.concatenate(vp7); vy7 = np.concatenate(vy7)
    try:    val_auc = roc_auc_score(vy7, vp7)
    except Exception: val_auc = float("nan")

    train_losses.append(running / len(tr_loader.dataset))
    val_losses.append(vrun / len(val_loader.dataset))
    print(f"  Epoch {epoch:2d}/{EPOCHS}  train={train_losses[-1]:.4f}  "
          f"val={val_losses[-1]:.4f}  val_auc7={val_auc:.3f}")

    if not np.isnan(val_auc) and val_auc > best_auc:
        best_auc = val_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        stale = 0
    else:
        stale += 1
        if stale >= EARLY_STOP_PATIENCE:
            print(f"   ⏹ early stop (no val AUC improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs)")
            break

if best_state is not None:
    model.load_state_dict(best_state)
print(f"✅ Best val ROC-AUC (t+7): {best_auc:.3f}")

# Loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label="train"); ax.plot(val_losses, label="val")
ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss (sum of t+7 & t+30)")
ax.set_title("Training Loss Curve"); ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/training_loss.png", dpi=150, bbox_inches="tight")
plt.show()

torch.save(model.state_dict(), f"{PLOT_DIR}/lce_predictor.pt")
print(f"✅ Model weights → {PLOT_DIR}/lce_predictor.pt")

try:
    import mlflow
    mlflow.start_run(run_name="gulf-watch-ibtracs-supervised")
    mlflow.log_params({
        "SEQ_LEN": SEQ_LEN, "HIDDEN_SIZE": HIDDEN_SIZE, "DROPOUT": DROPOUT,
        "EPOCHS": EPOCHS, "USE_CNN": USE_CNN,
        "label_source": "ibtracs_gulf_ri_onset",
    })
    mlflow.log_metric("val_roc_auc_best", best_auc)
    mlflow.end_run()
    print("✅ MLflow run logged")
except Exception:
    print("   (MLflow skipped)")

# ✅ DEMO CHECKPOINT: safe stopping point if time runs out


# CELL 8 — EVALUATION & KATRINA SANITY CHECK
# =============================================================================

model.eval()
p7_all, p30_all = [], []
with torch.no_grad():
    for xb, _, _ in val_loader:
        a, b = model(xb.to(device))
        p7_all.append(a.cpu().numpy()); p30_all.append(b.cpu().numpy())
p7_pred  = np.concatenate(p7_all)
p30_pred = np.concatenate(p30_all)

def _safe_auc(y, p):
    try:    return roc_auc_score(y, p)
    except: return float("nan")

auc7  = _safe_auc(y7_val,  p7_pred)
auc30 = _safe_auc(y30_val, p30_pred)
f1_7  = f1_score(y7_val,  (p7_pred  > 0.5).astype(int), zero_division=0)
f1_30 = f1_score(y30_val, (p30_pred > 0.5).astype(int), zero_division=0)
print(f"✅ Validation:")
print(f"   t+7  :  AUC-ROC={auc7:.3f}   F1={f1_7:.3f}")
print(f"   t+30 :  AUC-ROC={auc30:.3f}   F1={f1_30:.3f}")

# 🎯 KATRINA CHECK — predict separation probability for 90 days before Aug 28 2005
katrina_date = pd.Timestamp("2005-08-28")
window_start = katrina_date - pd.Timedelta(days=90)
mask = (lc_series.index >= window_start) & (lc_series.index <= katrina_date + pd.Timedelta(days=7))
window_idx = np.where(np.asarray(mask))[0]

katrina_probs = []
model.eval()
with torch.no_grad():
    for t_idx in window_idx:
        if t_idx < SEQ_LEN or t_idx + 30 >= T: continue
        x = torch.from_numpy(feats[t_idx - SEQ_LEN:t_idx]).unsqueeze(0).to(device)
        p7_k, p30_k = model(x)
        katrina_probs.append((lc_series.index[t_idx],
                              float(p7_k.item()), float(p30_k.item())))

if katrina_probs:
    kdf = pd.DataFrame(katrina_probs, columns=["date", "p7", "p30"]).set_index("date")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(kdf.index, kdf["p7"], label="P(separation at t+7)", color="tomato", lw=2)
    ax.plot(kdf.index, kdf["p30"], label="P(separation at t+30)", color="steelblue",
            lw=2, ls="--")
    ax.axvline(katrina_date, color="purple", lw=2, label="Katrina Aug 28")
    ax.axhline(0.5, color="gray", ls=":")
    ax.set_ylim(0, 1)
    ax.set_title("🎯 Katrina Sanity Check — LCE Separation Probability (90 days pre-landfall)")
    ax.set_xlabel("Date"); ax.set_ylabel("Probability")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/katrina_separation_prob.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Aug 25 point estimates
    target = pd.Timestamp("2005-08-25")
    if target in kdf.index:
        row = kdf.loc[target]
    else:
        nearest = kdf.index[np.argmin(np.abs(kdf.index - target))]
        row = kdf.loc[nearest]
        print(f"   (nearest available: {nearest.date()})")
    print(f"✅ Aug 25 2005:  P(t+7)={row['p7']:.3f}   P(t+30)={row['p30']:.3f}")
else:
    print("⚠️ Katrina date is inside training window — skipping pre-landfall plot")

# ✅ DEMO CHECKPOINT: safe stopping point if time runs out


# CELL 9 — RISK HEATMAP (baseline)
# =============================================================================

def risk_heatmap(date_str, warming_offset=0.0, save_path=None):
    frame, actual = _nearest_date(ssh_anomaly, date_str)
    field = np.nan_to_num(frame.values, nan=0.0) + warming_offset

    ssh_max = max(float(np.abs(field).max()), 1e-6)
    risk = np.clip(field / ssh_max, 0, 1)
    risk = gaussian_filter(risk, sigma=1.2)

    lons_arr = frame[lon_dim].values
    lats_arr = frame[lat_dim].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im0 = axes[0].pcolormesh(lons_arr, lats_arr, field,
                             cmap="RdBu_r", vmin=-0.4, vmax=0.4, shading="auto")
    axes[0].contour(lons_arr, lats_arr, field,
                    levels=[RI_THRESHOLD], colors="white", linewidths=2, linestyles="--")
    axes[0].set_title(f"SSH Anomaly  —  {actual}" +
                      (f"  (+{warming_offset:.2f}m warming)" if warming_offset else ""))
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
    plt.colorbar(im0, ax=axes[0], label="SSH Anomaly (m)")

    im1 = axes[1].pcolormesh(lons_arr, lats_arr, risk,
                             cmap="YlOrRd", vmin=0, vmax=1, shading="auto")
    axes[1].contour(lons_arr, lats_arr, field,
                    levels=[RI_THRESHOLD], colors="white", linewidths=2, linestyles="--")
    axes[1].set_title(f"RI Risk (0–1)  —  {actual}")
    axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[1], label="Risk (0–1)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return field, risk, actual

field_base, risk_base, _ = risk_heatmap("2005-08-25",
                                        save_path=f"{PLOT_DIR}/risk_heatmap_20050825.png")
print(f"✅ Baseline risk heatmap → {PLOT_DIR}/risk_heatmap_20050825.png")
base_hot_pct = 100 * np.mean(risk_base > 0.5)
print(f"   Gulf area with risk > 0.5 : {base_hot_pct:.1f}%")

# ✅ DEMO CHECKPOINT: safe stopping point if time runs out


# CELL 10 — RISK HEATMAP (+2°C WARMING SCENARIO)
# =============================================================================

field_warm, risk_warm, _ = risk_heatmap("2005-08-25",
                                        warming_offset=WARMING_OFFSET,
                                        save_path=f"{PLOT_DIR}/risk_heatmap_20050825_warm.png")
warm_hot_pct = 100 * np.mean(risk_warm > 0.5)
print(f"✅ +2°C scenario risk heatmap → {PLOT_DIR}/risk_heatmap_20050825_warm.png")
print(f"   Gulf area with risk > 0.5 : {warm_hot_pct:.1f}%   (Δ = +{warm_hot_pct - base_hot_pct:.1f} pp)")

# Summary JSON for dashboard/stub calibration
summary = {
    "built": str(datetime.utcnow()),
    "ssh_var": ssh_var,
    "use_cnn": USE_CNN,
    "time_range": [str(ssh_gulf[time_dim].values[0])[:10],
                   str(ssh_gulf[time_dim].values[-1])[:10]],
    "gulf_bbox":   {"lat": [LAT_MIN, LAT_MAX], "lon": [LON_MIN, LON_MAX]},
    "lc_zone":     {"lat": [LC_LAT_MIN, LC_LAT_MAX], "lon": [LC_LON_MIN, LC_LON_MAX]},
    "ri_threshold_m": RI_THRESHOLD,
    "warming_offset_m": WARMING_OFFSET,
    "val_auc_t7":  round(float(globals().get("auc7",  best_auc)), 3),
    "val_auc_t30": round(float(globals().get("auc30", float("nan"))), 3),
    "val_f1_t7":   round(float(globals().get("f1_7",  float("nan"))), 3),
    "val_f1_t30":  round(float(globals().get("f1_30", float("nan"))), 3),
    "baseline_hot_pct": round(float(base_hot_pct), 2),
    "warm_hot_pct":     round(float(warm_hot_pct), 2),
}
with open(f"{PLOT_DIR}/gulf_watch_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"✅ Summary → {PLOT_DIR}/gulf_watch_summary.json")
print(json.dumps(summary, indent=2))

print("\n" + "=" * 60)
print("🏁 GULF WATCH NOTEBOOK COMPLETE")
print("=" * 60)
print("Artefacts in /tmp/:")
print("  anchor_katrina.png / anchor_harvey.png / anchor_ida.png")
print("  separation_labels.png  training_loss.png")
print("  katrina_separation_prob.png")
print("  risk_heatmap_20050825.png")
print("  risk_heatmap_20050825_warm.png")
print("  lce_predictor.pt  gulf_watch_summary.json")

# ✅ DEMO CHECKPOINT: safe stopping point if time runs out


# CELL 11 — IBTrACS EXTERNAL VALIDATION (honest metrics)
# =============================================================================
# Honest evaluation against NOAA IBTrACS best-track observations.
#   A) Run model across the full time series → one probability per day.
#   B) Build a ground-truth vector: 1 if a Gulf RI onset occurs within
#      HIT_LEAD_DAYS after that day, else 0. Restrict to Jun–Nov.
#   C) Pick a threshold on the VAL split (chronological, same as Cell 7).
#   D) Report TP / FP / FN / TN + precision, recall, false-alarm rate on the
#      full hurricane-season vector using that fixed threshold.
#   E) Compare against the dumb "SSH anomaly > 17 cm" baseline on the same
#      ground truth. If the baseline matches the model, the LSTM isn't adding
#      value.
#   F) Per-storm plots kept for the deck.

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

IBTRACS_PATH = "/Volumes/workspace/default/gulf_watch/ibtracs.NA.list.v04r01.csv"

RI_WIND_DELTA_KT = 30.0
RI_WINDOW_HOURS  = 24
GULF_BBOX        = {"lat": (18.0, 32.0), "lon": (-98.0, -78.0)}
HIT_LEAD_DAYS    = 7
HURR_MONTHS      = [6, 7, 8, 9, 10, 11]

VALIDATION_STORMS = [
    ("KATRINA", 2005), ("RITA", 2005), ("HARVEY", 2017),
    ("IDA", 2021),    ("MICHAEL", 2018),
]

assert os.path.exists(IBTRACS_PATH), f"Missing {IBTRACS_PATH}"
print(f"Loading IBTrACS from {IBTRACS_PATH} …")
ib = pd.read_csv(IBTRACS_PATH, skiprows=[1], low_memory=False,
                 na_values=[" ", "", "NA"])
ib["ISO_TIME"] = pd.to_datetime(ib["ISO_TIME"], errors="coerce")
for _c in ("LAT", "LON", "USA_WIND", "WMO_WIND"):
    if _c in ib.columns:
        ib[_c] = pd.to_numeric(ib[_c], errors="coerce")
ib["WIND_KT"] = ib["USA_WIND"].fillna(ib["WMO_WIND"])
ib = ib[(ib["SEASON"] >= 1982)].dropna(subset=["WIND_KT", "ISO_TIME", "LAT", "LON"])

# Restrict to Gulf-of-Mexico track points only
gulf = ib[ib["LAT"].between(*GULF_BBOX["lat"]) &
          ib["LON"].between(*GULF_BBOX["lon"])].copy()
print(f"   Gulf track points since 1982: {len(gulf)}")

def _flag_ri(g):
    g = g.sort_values("ISO_TIME").reset_index(drop=True)
    t = g["ISO_TIME"].values
    w = g["WIND_KT"].values
    flag = np.zeros(len(g), dtype=bool)
    for i in range(len(g)):
        dt_h = (t - t[i]).astype("timedelta64[h]").astype(float)
        later = np.where(dt_h >= RI_WINDOW_HOURS)[0]
        if len(later) == 0:
            break
        j = later[0]
        if w[j] - w[i] >= RI_WIND_DELTA_KT:
            flag[i] = True
    return g.assign(RI_FLAG=flag)

gulf = gulf.groupby("SID", group_keys=False).apply(_flag_ri)
ri_onset_days = pd.DatetimeIndex(
    pd.to_datetime(gulf.loc[gulf["RI_FLAG"], "ISO_TIME"]).dt.floor("D").unique()
).sort_values()
print(f"   Gulf RI onset days (1982+): {len(ri_onset_days)}")

# ------- A) Full-timeseries model inference (batched, memory-safe) -------
import gc
model.eval()
idxs = list(range(SEQ_LEN, T - 30))
probs7 = np.zeros(len(idxs), dtype=np.float32)
probs30 = np.zeros(len(idxs), dtype=np.float32)
BATCH = 32
with torch.no_grad():
    for start in range(0, len(idxs), BATCH):
        end = min(start + BATCH, len(idxs))
        xs = np.stack([feats[idxs[k] - SEQ_LEN:idxs[k]] for k in range(start, end)])
        p7_v, p30_v = model(torch.from_numpy(xs).to(device))
        probs7[start:end] = p7_v.cpu().numpy()
        probs30[start:end] = p30_v.cpu().numpy()
        del xs, p7_v, p30_v
gc.collect()

prob_df = pd.DataFrame(
    {"p7": probs7, "p30": probs30},
    index=pd.DatetimeIndex([lc_series.index[i] for i in idxs]),
)
print(f"   Full-timeseries probs: n={len(prob_df)}")

# ------- B) Ground truth over hurricane-season days -------
hurr = prob_df[prob_df.index.month.isin(HURR_MONTHS)].copy()
onset_arr = ri_onset_days.values.astype("datetime64[D]")
days = hurr.index.values.astype("datetime64[D]")
y_true = np.zeros(len(days), dtype=int)
for k, d in enumerate(days):
    dt = (onset_arr - d).astype("timedelta64[D]").astype(int)
    if ((dt >= 0) & (dt <= HIT_LEAD_DAYS)).any():
        y_true[k] = 1
hurr["y_true"] = y_true
print(f"   Positive days (RI onset within {HIT_LEAD_DAYS}d): "
      f"{int(hurr['y_true'].sum())} / {len(hurr)} ({hurr['y_true'].mean():.2%})")

# ------- C) Threshold picked on VAL split only (no peeking) -------
# Recompute the same chronological split as Cell 7 (38/42 years → train/val).
_split = int(len(idxs) * (38.0 / 42.0))
val_dates = pd.DatetimeIndex([lc_series.index[idxs[i]] for i in range(_split, len(idxs))])
val_mask  = hurr.index.isin(val_dates)
val_df    = hurr[val_mask]
print(f"   Val hurricane-season days: {len(val_df)}  "
      f"(positives: {int(val_df['y_true'].sum())})")

if val_df["y_true"].sum() == 0 or len(val_df) < 10:
    print("   ⚠ Val split has too few positives — falling back to threshold=0.5")
    THRESHOLD = 0.5
    val_auc = val_ap = float("nan")
else:
    prec, rec, thr = precision_recall_curve(val_df["y_true"], val_df["p7"])
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    best = int(np.argmax(f1[:-1])) if len(thr) else 0
    THRESHOLD = float(thr[best]) if len(thr) else 0.5
    val_auc = float(roc_auc_score(val_df["y_true"], val_df["p7"]))
    val_ap  = float(average_precision_score(val_df["y_true"], val_df["p7"]))
    print(f"   Chosen threshold (val F1-max): {THRESHOLD:.3f}  "
          f"P={prec[best]:.2f}  R={rec[best]:.2f}  F1={f1[best]:.2f}")
    print(f"   Val ROC-AUC={val_auc:.3f}   PR-AP={val_ap:.3f}")

# ------- D) Global confusion matrix on the full hurricane-season vector -------
def _confusion(y, scores, thr):
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return dict(
        TP=tp, FP=fp, FN=fn, TN=tn,
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        false_alarm_rate=fp / max(fp + tn, 1),
    )

model_cm = _confusion(hurr["y_true"].values, np.asarray(hurr["p7"].values), THRESHOLD)
print(f"\n🎯 Model (p7 ≥ {THRESHOLD:.3f}) — full hurricane-season eval:")
print(f"   TP={model_cm['TP']}  FP={model_cm['FP']}  "
      f"FN={model_cm['FN']}  TN={model_cm['TN']}")
print(f"   Precision={model_cm['precision']:.2%}  "
      f"Recall={model_cm['recall']:.2%}  FAR={model_cm['false_alarm_rate']:.2%}")

# ------- E) SSH-only baseline: does the LSTM beat a single threshold? -------
# Fraction of Gulf area where SSH anomaly > 17 cm (Mainelli threshold).
# Max-pixel saturates because the Loop Current is a permanent warm bulge;
# the *area* of the anomaly varies day-to-day and is the real signal.
# Memory-safe: reduce straight on the bool array (no float cast) so we never
# materialise a full (T, H, W) float copy.
_area_frac_daily = (ssh_anomaly > RI_THRESHOLD).mean(dim=[lat_dim, lon_dim])
_area_series = pd.Series(
    np.asarray(_area_frac_daily.values, dtype=np.float32),
    index=pd.to_datetime(_area_frac_daily[time_dim].values),
)
del _area_frac_daily
area_today = _area_series.reindex(hurr.index).values

# Tune the baseline threshold on val split (same no-peeking rule as the model)
base_val = area_today[val_mask]
if val_df["y_true"].sum() > 0 and len(base_val) >= 10:
    b_prec, b_rec, b_thr = precision_recall_curve(val_df["y_true"], base_val)
    b_f1 = 2 * b_prec * b_rec / np.maximum(b_prec + b_rec, 1e-9)
    b_best = int(np.argmax(b_f1[:-1])) if len(b_thr) else 0
    BASE_THRESHOLD = float(b_thr[b_best]) if len(b_thr) else float(np.median(base_val))
else:
    BASE_THRESHOLD = float(np.median(base_val))

base_pred = (area_today >= BASE_THRESHOLD).astype(int)
base_cm = _confusion(hurr["y_true"].values, base_pred.astype(float), 0.5)
print(f"\n📏 Baseline (Gulf area fraction > {RI_THRESHOLD} m "
      f"≥ {BASE_THRESHOLD:.3f}):")
print(f"   TP={base_cm['TP']}  FP={base_cm['FP']}  "
      f"FN={base_cm['FN']}  TN={base_cm['TN']}")
print(f"   Precision={base_cm['precision']:.2%}  "
      f"Recall={base_cm['recall']:.2%}  FAR={base_cm['false_alarm_rate']:.2%}")

delta = {k: model_cm[k] - base_cm[k]
         for k in ("precision", "recall", "false_alarm_rate")}
print(f"\n🔬 Model − Baseline:  ΔP={delta['precision']:+.2%}  "
      f"ΔR={delta['recall']:+.2%}  ΔFAR={delta['false_alarm_rate']:+.2%}")

# Persist
with open(f"{PLOT_DIR}/ibtracs_validation.json", "w") as f:
    json.dump({
        "hit_lead_days": HIT_LEAD_DAYS,
        "threshold": THRESHOLD,
        "val_roc_auc": val_auc,
        "val_pr_ap": val_ap,
        "model": model_cm,
        "ssh_baseline": base_cm,
        "gulf_ri_onset_days": int(len(ri_onset_days)),
        "hurricane_season_days_evaluated": int(len(hurr)),
    }, f, indent=2, default=str)
print(f"\n💾 Summary → {PLOT_DIR}/ibtracs_validation.json")

# ------- F) Per-storm plots for the deck (no longer the metric source) -------
for name, year in VALIDATION_STORMS:
    track = gulf[(gulf["NAME"].str.upper() == name) &
                 (gulf["SEASON"] == year)].sort_values("ISO_TIME")
    if len(track) == 0:
        print(f"   ⚠ {name} {year}: no Gulf track record")
        continue
    onsets = pd.to_datetime(track.loc[track["RI_FLAG"], "ISO_TIME"])
    peak = pd.Timestamp(track.loc[track["WIND_KT"].idxmax(), "ISO_TIME"])
    win = ((prob_df.index >= peak - pd.Timedelta(days=60)) &
           (prob_df.index <= peak + pd.Timedelta(days=3)))
    pdf = prob_df.loc[win]
    if pdf.empty:
        print(f"   ⚠ {name} {year}: outside model range")
        continue

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    a1.plot(pdf.index, pdf["p7"], color="tomato", lw=2, label="P(sep t+7)")
    a1.plot(pdf.index, pdf["p30"], color="steelblue", lw=1.5, ls="--",
            label="P(sep t+30)")
    a1.axhline(THRESHOLD, color="gray", ls=":", lw=1,
               label=f"threshold {THRESHOLD:.2f}")
    for o in onsets:
        a1.axvline(pd.Timestamp(o), color="red", lw=1.2, alpha=0.6)
    a1.set_ylim(0, 1); a1.set_ylabel("Probability")
    a1.set_title(f"{name.title()} ({year}) — Model vs IBTrACS",
                 fontsize=12, fontweight="bold")
    a1.legend(loc="upper left", fontsize=9); a1.grid(alpha=0.25)

    a2.plot(track["ISO_TIME"], track["WIND_KT"], color="orange", lw=2,
            label="IBTrACS wind (kt)")
    a2.scatter(onsets, track.loc[track["RI_FLAG"], "WIND_KT"],
               color="red", marker="^", s=70, zorder=5,
               label="RI onset (≥ 30 kt / 24 h)")
    a2.axhline(64, color="#fde68a", ls=":", lw=0.7)
    a2.axhline(113, color="#ef4444", ls=":", lw=0.7)
    a2.set_ylabel("Wind (kt)"); a2.set_xlabel("Date (UTC)")
    a2.legend(loc="upper left", fontsize=9); a2.grid(alpha=0.25)

    plt.tight_layout()
    out = f"{PLOT_DIR}/validation_{name.lower()}_{year}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"   ✅ {name} {year}: {len(onsets)} onset(s) → {out}")

# ✅ DEMO CHECKPOINT: safe stopping point if time runs out
