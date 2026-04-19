# =============================================================================
# GULF WATCH — Databricks Model Serving wrapper
# =============================================================================
# NOTE: Databricks Free Edition disables the model registry, so the REGISTER +
# DEPLOY cell below cannot run there. For that case, use `server.py` in this
# directory — it runs the same inference pipeline as a local FastAPI process.
# Keep this file around for when you move to a paid workspace.
#
# Wraps the trained CNN+LSTM model in an MLflow pyfunc so it can be served
# from a Databricks Model Serving endpoint and called from the Next.js app
# via /api/predict.
#
# Workflow:
#   1. Run gulf_watch_notebook.py through Cell 7 → produces lce_predictor.pt.
#   2. Run "BAKE ARTIFACTS" cell below in the same notebook session — it has
#      access to ssh_anomaly, std_ch, etc. from the training run.
#   3. Run "REGISTER + DEPLOY" cell below — logs the pyfunc to MLflow and
#      prints the endpoint URL to put in .env.local.
#
# The endpoint accepts the CLAUDE.md REST contract:
#   {"date": "YYYY-MM-DD", "sst_delta": float, "loop_depth": float}
# →
#   {"ri_probability": [[float]], "lce_separation_prob_7d": float,
#    "lce_separation_prob_30d": float, "ri_days_per_year": float,
#    "highest_risk_zone": str}
# =============================================================================


# CELL A — BAKE ARTIFACTS (run after Cell 7 of the training notebook)
# =============================================================================
# Expects in scope: raw, std_ch, ssh_anomaly, lat_dim, lon_dim, time_dim,
#                   _lag, H, W, model

import os, json, shutil
import numpy as np

ARTIFACTS_DIR = "/tmp/gulf_watch_serving"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# NOTE (free-tier pivot): ssh_anomaly.npz is no longer baked here — it's too
# large for Workspace Files on Free Edition. Instead, run
# `python databricks/preprocess_local.py /path/to/run2_clim_v2_ssh.nc`
# on your laptop to produce artifacts/ssh_anomaly.npz there.
# Only the tiny artifacts below need to come from Databricks.

# 1. Copy the trained weights.
shutil.copyfile("/tmp/lce_predictor.pt",
                os.path.join(ARTIFACTS_DIR, "lce_predictor.pt"))

# 2. Save metadata JSON. std_ch is included here (not in the npz) so
#    preprocess_local.py can produce the npz independently of the training run.
with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w") as f:
    json.dump({
        "input_size": int(model.lstm.input_size),
        "hidden_size": int(model.lstm.hidden_size),
        "h": int(H),
        "w": int(W),
        "lag": int(_lag),
        "use_cnn": bool(USE_CNN),
        "std_ch": std_ch.tolist(),
    }, f, indent=2)

print(f"✅ Baked serving artifacts → {ARTIFACTS_DIR}")
print("   ", os.listdir(ARTIFACTS_DIR))


# CELL B — PYFUNC WRAPPER + REGISTER + DEPLOY
# =============================================================================

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import torch
import json
import os

# Local import — the lib lives next to this file.
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) or ".")
from gulf_watch_lib import (
    SSHEncoder,
    LCEPredictor,
    SEQ_LEN,
    RI_THRESHOLD,
    WARMING_OFFSET,
    build_three_channels,
    risk_field_from_anomaly,
    label_region,
)


class GulfWatchModel(mlflow.pyfunc.PythonModel):
    """Date-driven CNN+LSTM RI predictor for Databricks Model Serving."""

    def load_context(self, context):
        meta = json.loads(open(context.artifacts["metadata"]).read())
        self.h, self.w = meta["h"], meta["w"]
        self.input_size = meta["input_size"]
        self.lag = meta["lag"]
        self.use_cnn = meta["use_cnn"]

        npz = np.load(context.artifacts["anomaly"], allow_pickle=False)
        self.raw = npz["raw"]                      # (T, H, W) float32
        self.times = pd.DatetimeIndex(npz["times"])  # frame dates
        self.lats = npz["lats"]
        self.lons = npz["lons"]
        self.std_ch = npz["std_ch"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Frozen random-init CNN encoder — matches training initialization
        # via torch.manual_seed(42) at the top of the notebook. Re-seed here.
        torch.manual_seed(42)
        self.encoder = SSHEncoder(self.h, self.w).to(device).eval()

        self.model = LCEPredictor(input_size=self.input_size).to(device)
        self.model.load_state_dict(
            torch.load(context.artifacts["weights"], map_location=device)
        )
        self.model.eval()

    def _date_to_index(self, date_str: str) -> int:
        target = pd.Timestamp(date_str)
        return int(np.argmin(np.abs(self.times - target)))

    def _features_for_window(self, end_idx: int) -> np.ndarray:
        """Build the (SEQ_LEN, input_size) feature sequence ending at end_idx."""
        start = max(0, end_idx - SEQ_LEN)
        # Pad at the left if too close to series start
        if end_idx - start < SEQ_LEN:
            pad = SEQ_LEN - (end_idx - start)
            window = np.concatenate(
                [np.repeat(self.raw[start:start + 1], pad, axis=0),
                 self.raw[start:end_idx]],
                axis=0,
            )
            actual_start = 0
        else:
            window = self.raw[start:end_idx]
            actual_start = start

        if not self.use_cnn:
            # Fallback: compute scalar features per day from the LC zone mean.
            # The training-time scalar features are ~zero-mean by construction.
            lc = window.mean(axis=(1, 2))
            return np.stack([lc, lc, lc], axis=1).astype(np.float32)

        # CNN feature extraction — same channel builder as training.
        chans = build_three_channels(
            self.raw, actual_start, actual_start + SEQ_LEN,
            self.std_ch, self.lag,
        )[:SEQ_LEN]
        # If we padded, recompute the head on the truncated raw
        if window.shape[0] != self.raw[actual_start:actual_start + SEQ_LEN].shape[0]:
            # Edge case: very early dates. Just use the available window.
            chans = chans[-window.shape[0]:]
            if chans.shape[0] < SEQ_LEN:
                chans = np.concatenate(
                    [np.repeat(chans[:1], SEQ_LEN - chans.shape[0], axis=0), chans],
                    axis=0,
                )
        with torch.no_grad():
            feats = self.encoder(
                torch.from_numpy(chans).to(self.device)
            ).cpu().numpy()
        return feats.astype(np.float32)

    def _predict_one(self, date: str, sst_delta: float, loop_depth: float):
        idx = self._date_to_index(date)

        # 1. Model scalars (p7, p30) from the 30-day window ending at this date.
        feats = self._features_for_window(idx)
        with torch.no_grad():
            x = torch.from_numpy(feats).unsqueeze(0).to(self.device)
            p7, p30 = self.model(x)
            p7, p30 = float(p7.item()), float(p30.item())

        # 2. Heatmap from the SSH anomaly frame at this date, with SST warming
        #    proxy applied (notebook Cell 10 logic, generalised to a continuous
        #    sst_delta rather than the binary +2°C scenario).
        warming_m = sst_delta * (WARMING_OFFSET / 2.0)
        field = self.raw[idx] + warming_m
        risk = risk_field_from_anomaly(field)

        # 3. Annualised ri_days_per_year — count of cells above RI threshold
        #    in the 365-day window centred on this date, normalised to per-year.
        win_lo = max(0, idx - 182)
        win_hi = min(self.raw.shape[0], idx + 183)
        annual_window = self.raw[win_lo:win_hi] + warming_m
        # mean fraction of pixels above threshold × 365 ≈ days per year
        ri_days = float(np.mean(annual_window > RI_THRESHOLD) * 365.0)

        # 4. Highest-risk zone label from argmax of the risk grid.
        peak_j, peak_i = np.unravel_index(int(np.argmax(risk)), risk.shape)
        zone = label_region(float(self.lats[peak_j]), float(self.lons[peak_i]))

        return {
            "ri_probability": risk.astype(float).tolist(),
            "lce_separation_prob_7d": p7,
            "lce_separation_prob_30d": p30,
            "ri_days_per_year": ri_days,
            "highest_risk_zone": zone,
        }

    def predict(self, context, model_input):
        # model_input is a pandas DataFrame with one row when called via REST.
        if isinstance(model_input, pd.DataFrame):
            row = model_input.iloc[0]
            return [self._predict_one(
                str(row["date"]),
                float(row["sst_delta"]),
                float(row["loop_depth"]),
            )]
        # Direct dict-style call (e.g. for local smoke testing)
        return self._predict_one(
            model_input["date"], model_input["sst_delta"], model_input["loop_depth"],
        )


# --- Register to MLflow + Unity Catalog -------------------------------------

ARTIFACTS_DIR = "/tmp/gulf_watch_serving"

example_input = pd.DataFrame([{
    "date": "2005-08-25",
    "sst_delta": 0.0,
    "loop_depth": 0.5,
}])

with mlflow.start_run(run_name="gulf-watch-serving"):
    info = mlflow.pyfunc.log_model(
        artifact_path="gulf_watch",
        python_model=GulfWatchModel(),
        artifacts={
            "weights": os.path.join(ARTIFACTS_DIR, "lce_predictor.pt"),
            "anomaly": os.path.join(ARTIFACTS_DIR, "ssh_anomaly.npz"),
            "metadata": os.path.join(ARTIFACTS_DIR, "metadata.json"),
        },
        code_path=["gulf_watch_lib.py"],
        pip_requirements=[
            "torch", "numpy", "pandas", "scipy", "scikit-learn", "mlflow",
        ],
        input_example=example_input,
    )
    run_id = mlflow.active_run().info.run_id

print(f"✅ Logged model: runs:/{run_id}/gulf_watch")

# Register to Unity Catalog. Adjust the catalog/schema to match your workspace.
CATALOG = "workspace"
SCHEMA = "default"
MODEL_NAME = "gulf_watch_v1"
mv = mlflow.register_model(
    model_uri=f"runs:/{run_id}/gulf_watch",
    name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
)
print(f"✅ Registered: {CATALOG}.{SCHEMA}.{MODEL_NAME} v{mv.version}")

# --- DEPLOY -----------------------------------------------------------------
# After this script runs, in the Databricks UI:
#   1. Serving → Create endpoint → name "gulf-watch-v1"
#   2. Pick the registered model + version above
#   3. Compute: CPU Small (workload size = small, scale-to-zero on)
#   4. After it goes Ready, copy the invocation URL — looks like:
#      https://<workspace>.cloud.databricks.com/serving-endpoints/gulf-watch-v1/invocations
#   5. Generate a PAT under user settings → Developer → Access tokens
#   6. Put both into the Next.js app's .env.local:
#      DATABRICKS_ENDPOINT_URL=...
#      DATABRICKS_TOKEN=dapi...
#
# Smoke test from a notebook cell:
#   import requests, os
#   r = requests.post(
#       os.environ["DATABRICKS_ENDPOINT_URL"],
#       headers={"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"},
#       json={"dataframe_records": [{"date":"2005-08-25","sst_delta":0,"loop_depth":0.5}]},
#   )
#   print(r.status_code, r.json())
#
# Expect p7 > 0.5 for the Katrina date, per the in-notebook sanity check.
