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
    """Encodes a single 2D SSH-anomaly field into a 128-dim feature vector."""
    def __init__(self, h, w):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 1, h, w)).view(1, -1).shape[1]
        self.fc = nn.Linear(flat, 128)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dynamic input shape from actual cropped grid
sample = ssh_anomaly.isel({time_dim: 0}).values
H, W = sample.shape
print(f"✅ Grid size: H={H}  W={W}  (device={device})")

# Build daily feature matrix: shape (T, 128) if USE_CNN, else (T, 3)
if USE_CNN:
    print("Encoding 2D SSH-anomaly fields with CNN …")
    encoder = SSHEncoder(H, W).to(device).eval()
    arr = np.nan_to_num(ssh_anomaly.values, nan=0.0).astype(np.float32)

    feats = np.zeros((arr.shape[0], 128), dtype=np.float32)
    step = 256
    with torch.no_grad():
        for i in range(0, arr.shape[0], step):
            batch = torch.from_numpy(arr[i:i+step]).unsqueeze(1).to(device)
            feats[i:i+step] = encoder(batch).cpu().numpy()
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


# CELL 7 — LSTM PREDICTOR & TRAINING
# =============================================================================

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
        return torch.sigmoid(self.fc7(last)).squeeze(-1), \
               torch.sigmoid(self.fc30(last)).squeeze(-1)

# Build (X, y7, y30) sequences
T = feats.shape[0]
labels = sep_label.values.astype(np.float32)

X_list, y7_list, y30_list = [], [], []
for i in range(SEQ_LEN, T - 30):
    X_list.append(feats[i - SEQ_LEN : i])
    y7_list.append(labels[i + 7])
    y30_list.append(labels[i + 30])

X_all  = np.asarray(X_list, dtype=np.float32)
y7_all = np.asarray(y7_list, dtype=np.float32)
y30_all = np.asarray(y30_list, dtype=np.float32)
print(f"✅ Sequences: X={X_all.shape}  y7={y7_all.shape}  y30={y30_all.shape}")

# Dynamic 38yr / remainder split
split = int(len(X_all) * (38.0 / 42.0))
X_tr, X_val = X_all[:split], X_all[split:]
y7_tr, y7_val = y7_all[:split], y7_all[split:]
y30_tr, y30_val = y30_all[:split], y30_all[split:]
print(f"   Train={len(X_tr)}  Val={len(X_val)}")

tr_loader = DataLoader(
    TensorDataset(torch.tensor(X_tr), torch.tensor(y7_tr), torch.tensor(y30_tr)),
    batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val), torch.tensor(y7_val), torch.tensor(y30_val)),
    batch_size=BATCH_SIZE, shuffle=False)

model = LCEPredictor(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                     dropout=DROPOUT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss()

train_losses, val_losses = [], []
print(f"Training for {EPOCHS} epochs …")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for xb, y7b, y30b in tr_loader:
        xb = xb.to(device); y7b = y7b.to(device); y30b = y30b.to(device)
        p7, p30 = model(xb)
        loss = loss_fn(p7, y7b) + loss_fn(p30, y30b)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.item() * len(xb)

    model.eval(); vrun = 0.0
    with torch.no_grad():
        for xb, y7b, y30b in val_loader:
            xb = xb.to(device); y7b = y7b.to(device); y30b = y30b.to(device)
            p7, p30 = model(xb)
            vrun += (loss_fn(p7, y7b) + loss_fn(p30, y30b)).item() * len(xb)

    train_losses.append(running / len(tr_loader.dataset))
    val_losses.append(vrun / len(val_loader.dataset))
    if epoch % 2 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d}/{EPOCHS}  train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")

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

# ⚠️ OPTIONAL — remove if MLflow not configured
try:
    import mlflow
    mlflow.start_run(run_name="gulf-watch-lce-predictor")
    mlflow.log_params({"SEQ_LEN": SEQ_LEN, "HIDDEN_SIZE": HIDDEN_SIZE,
                       "DROPOUT": DROPOUT, "EPOCHS": EPOCHS, "USE_CNN": USE_CNN})
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
window_idx = np.where(mask.values)[0]

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
    "val_auc_t7":  round(float(auc7), 3),
    "val_auc_t30": round(float(auc30), 3),
    "val_f1_t7":   round(float(f1_7), 3),
    "val_f1_t30":  round(float(f1_30), 3),
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
