"""Local FastAPI replacement for the Databricks Model Serving endpoint.

Runs the same CNN+LSTM inference pipeline as `serving_wrapper.py` but without
MLflow or Databricks — use this on Databricks Free Edition where registry +
serving endpoints are disabled.

Run from repo root:
    GULF_WATCH_ARTIFACTS_DIR=./artifacts \
        uvicorn databricks.server:app --port 8000

Then point `.env.local` at this process:
    DATABRICKS_ENDPOINT_URL=http://localhost:8000/invocations
    DATABRICKS_TOKEN=local-dev      # ignored by this server

The request/response contract matches the Zod schemas in lib/databricks/types.ts
byte-for-byte so the Next.js client at lib/databricks/client.ts is oblivious
to the swap.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import gulf_watch_lib from the same directory — mirrors the sys.path pattern
# in serving_wrapper.py so this file can run whether invoked as
# `uvicorn databricks.server:app` from the repo root or directly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gulf_watch_lib import (  # noqa: E402
    SSHEncoder,
    LCEPredictor,
    SEQ_LEN,
    RI_THRESHOLD,
    WARMING_OFFSET,
    build_three_channels,
    risk_field_from_anomaly,
    label_region,
)


ARTIFACTS_DIR = Path(os.environ.get("GULF_WATCH_ARTIFACTS_DIR", "./artifacts"))


@dataclass
class ModelState:
    raw: np.ndarray            # (T, H, W) float32 SSH anomaly cube
    times: pd.DatetimeIndex    # frame dates, length T
    lats: np.ndarray           # (H,) float32
    lons: np.ndarray           # (W,) float32
    std_ch: np.ndarray         # (3,) per-channel std from training
    encoder: SSHEncoder
    model: LCEPredictor
    device: torch.device
    lag: int
    use_cnn: bool
    h: int
    w: int


STATE: ModelState | None = None


def _load_state(artifacts_dir: Path) -> ModelState:
    meta_path = artifacts_dir / "metadata.json"
    anomaly_path = artifacts_dir / "ssh_anomaly.npz"
    weights_path = artifacts_dir / "lce_predictor.pt"

    for p in (meta_path, anomaly_path, weights_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing artifact: {p}. "
                f"Bake artifacts via Cell A of serving_wrapper.py and copy "
                f"{artifacts_dir} from DBFS to this laptop."
            )

    meta = json.loads(meta_path.read_text())
    npz = np.load(anomaly_path, allow_pickle=False)
    raw = npz["raw"]
    times = pd.DatetimeIndex(npz["times"])
    lats = npz["lats"]
    lons = npz["lons"]
    # std_ch moved to metadata.json — npz is produced locally and omits it
    std_ch = np.array(meta["std_ch"], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The CNN encoder is frozen random-init — the training notebook seeds
    # torch.manual_seed(42) before constructing it, so we must do the same
    # here or the projection means something different on inference inputs.
    torch.manual_seed(42)
    encoder = SSHEncoder(meta["h"], meta["w"]).to(device).eval()

    model = LCEPredictor(input_size=meta["input_size"], hidden_size=meta["hidden_size"]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print(
        f"[gulf-watch] loaded raw shape (T, H, W) = {raw.shape}, "
        f"date range {times.min().date()} → {times.max().date()}, "
        f"device={device}",
        flush=True,
    )

    return ModelState(
        raw=raw,
        times=times,
        lats=lats,
        lons=lons,
        std_ch=std_ch,
        encoder=encoder,
        model=model,
        device=device,
        lag=int(meta["lag"]),
        use_cnn=bool(meta["use_cnn"]),
        h=int(meta["h"]),
        w=int(meta["w"]),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global STATE
    STATE = _load_state(ARTIFACTS_DIR)
    yield
    STATE = None


app = FastAPI(title="Gulf Watch — local serving", lifespan=lifespan)


# --- Request / response contract ------------------------------------------ #
# Mirrors lib/databricks/types.ts PredictRequestSchema byte-for-byte.

class PredictRequest(BaseModel):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    sst_delta: float = Field(ge=-1.0, le=3.0)
    loop_depth: float = Field(ge=0.0, le=1.0)


# --- Inference ------------------------------------------------------------ #

def _date_to_index(state: ModelState, date_str: str) -> int:
    target = pd.Timestamp(date_str)

    # TODO: out-of-range policy — see plan.
    # Currently clamps silently via argmin-distance: a date in 2030 snaps to
    # the latest frame in the dataset, a date in 1980 snaps to the earliest.
    # If you'd rather return an explicit signal to the UI, add an
    # `out_of_range` bool to the response dict here AND to PredictResponseSchema
    # in lib/databricks/types.ts. 5–10 lines.

    return int(np.argmin(np.abs(state.times - target)))


def _features_for_window(state: ModelState, end_idx: int) -> np.ndarray:
    """Build the (SEQ_LEN, input_size) feature sequence ending at end_idx."""
    start = max(0, end_idx - SEQ_LEN)
    if end_idx - start < SEQ_LEN:
        pad = SEQ_LEN - (end_idx - start)
        window = np.concatenate(
            [np.repeat(state.raw[start:start + 1], pad, axis=0),
             state.raw[start:end_idx]],
            axis=0,
        )
        actual_start = 0
    else:
        window = state.raw[start:end_idx]
        actual_start = start

    if not state.use_cnn:
        lc = window.mean(axis=(1, 2))
        return np.stack([lc, lc, lc], axis=1).astype(np.float32)

    chans = build_three_channels(
        state.raw, actual_start, actual_start + SEQ_LEN,
        state.std_ch, state.lag,
    )[:SEQ_LEN]
    if window.shape[0] != state.raw[actual_start:actual_start + SEQ_LEN].shape[0]:
        chans = chans[-window.shape[0]:]
        if chans.shape[0] < SEQ_LEN:
            chans = np.concatenate(
                [np.repeat(chans[:1], SEQ_LEN - chans.shape[0], axis=0), chans],
                axis=0,
            )
    with torch.no_grad():
        feats = state.encoder(
            torch.from_numpy(chans).to(state.device)
        ).cpu().numpy()
    return feats.astype(np.float32)


def predict_one(state: ModelState, date: str, sst_delta: float, loop_depth: float) -> dict:
    idx = _date_to_index(state, date)

    feats = _features_for_window(state, idx)
    with torch.no_grad():
        x = torch.from_numpy(feats).unsqueeze(0).to(state.device)
        p7, p30 = state.model(x)
        p7, p30 = float(p7.item()), float(p30.item())

    warming_m = sst_delta * (WARMING_OFFSET / 2.0)
    field = state.raw[idx] + warming_m
    risk = risk_field_from_anomaly(field)

    win_lo = max(0, idx - 182)
    win_hi = min(state.raw.shape[0], idx + 183)
    annual_window = state.raw[win_lo:win_hi] + warming_m
    ri_days = float(np.mean(annual_window > RI_THRESHOLD) * 365.0)

    peak_j, peak_i = np.unravel_index(int(np.argmax(risk)), risk.shape)
    zone = label_region(float(state.lats[peak_j]), float(state.lons[peak_i]))

    return {
        "ri_probability": risk.astype(float).tolist(),
        "lce_separation_prob_7d": p7,
        "lce_separation_prob_30d": p30,
        "ri_days_per_year": ri_days,
        "highest_risk_zone": zone,
    }


# --- Routes --------------------------------------------------------------- #

@app.get("/healthz")
def healthz() -> dict:
    return {"ok": STATE is not None}


@app.post("/invocations")
def invocations(req: PredictRequest) -> dict:
    if STATE is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return predict_one(STATE, req.date, req.sst_delta, req.loop_depth)
