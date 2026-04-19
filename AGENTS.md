# Gulf Watch

Hackathon project (Climate + AI/ML track, due **2026-04-18**). Compresses 40 years of Gulf of Mexico sea surface height (SSH) data into a real-time hurricane rapid-intensification (RI) prediction dashboard. Demo hook: drag an SST slider, watch the RI heatmap intensify and the events-per-year counter climb.

Full PRD: `~/Downloads/gulf_watch_general_prd.html`.

## Three-layer architecture

1. **Databricks (brain)** ## ML Model — CNN Encoder + LSTM (LCE Separation Predictor)

Two-stage spatial-temporal architecture:

**Stage 1 — CNN Encoder (SSHEncoder)**
- Input: single daily 2D SSH anomaly field (cropped Gulf grid, shape computed dynamically)
- 2 conv layers (8 and 16 filters, 3x3 kernels), ReLU, MaxPool2d after each
- Flattened and projected to 128-dim feature vector via Linear layer
- Input h/w must be derived from actual cropped grid shape — never hardcoded

**Stage 2 — LSTM Classifier (LCEPredictor)**
- Input: 30-day sequence of 128-dim CNN feature vectors
- Single LSTM layer, hidden size 64, dropout 0.2, batch_first=True
- Two output heads: fc7 (t+7 separation probability), fc30 (t+30 separation probability)
- Sigmoid activation on both outputs — binary classification, not regression

**Prediction target:** LCE separation state (binary: 1 = eddy detached/separating, 0 = attached/building)
**Labels:** Self-supervised from data — no external catalog. Separation flagged when LC zone 
SSH anomaly > 0.17m AND gradient between LC zone and western Gulf drops sharply. 
Smoothed with 5-day rolling window.

**Fallback:** Toggle USE_CNN = False to fall back to scalar LSTM on LC zone time series 
(input size 3: lc_ts, rolling_7, rolling_30). All downstream cells still run.

**Training:** PyTorch, BCELoss, Adam lr=1e-3, batch size 32, 20 epochs max
**Train split:** First 38 years of data (computed dynamically, not hardcoded to a year)
**Validate split:** Remaining ~4 years
2. **Marimo notebook (science UI)** — reactive notebook for Scripps judges. *Status: deferred until Databricks endpoint is live.*
3. **Dashboard UI (judge experience)** — Next.js + TypeScript + deck.gl. 5 panels: 40-year timelapse, hurricane scrubber, prediction heatmap, climate sliders, ElevenLabs voice agent. *Status: mockup done elsewhere; wire to API once ported in.*

## Dataset

**File:** run2_clim_v2_ssh.nc (~2GB, NetCDF format)  
**Source:** ECCO Gulf of Mexico State Estimation (Scripps / UCSD)  
**Nature:** 40-year daily numerical model simulation — NOT a climatological mean. 
"clim" in filename refers to the model run name.  
**Coverage:** ~1982–2022, daily frames (~14,600 total), full Gulf of Mexico spatial grid

**DBFS path (update to match upload location):**
`/dbfs/FileStore/gulf_watch/run2_clim_v2_ssh.nc`

**Variable name — handle all cases, detect at runtime:**
Possible names: ssh, eta, zos, sea_surface_height, SSH, ETA, adt
Use detection loop — never assume. Print all ds.data_vars if none found.

**Coordinate names — detect at runtime:**
- Latitude: lat, latitude, y, YC
- Longitude: lon, longitude, x, XC
- Time: time, TIME, t

**Longitude convention — check before cropping:**
May be stored as 0–360°E rather than −180–180°W.
If ds[lon_dim].max() > 180: use 263–280 for Gulf crop instead of -97 to -80.

**Memory — use chunked loading on Community Edition:**
`xr.open_dataset(path, chunks={time_dim: 365})`
Only call .load() and .values on the cropped Gulf subset, never the full dataset.
## Hard rules

- **Do NOT redesign the dashboard UI.** The mockup is the source of truth — wire it, don't rebuild it.
- **Databricks endpoint URL comes from `DATABRICKS_ENDPOINT_URL`.** Never hardcode. Token comes from `DATABRICKS_TOKEN`. Both are server-side only; never expose to the browser.
- **Stub fallback is mandatory.** If env vars are missing, the request fails, or the response fails schema validation, fall back to the stub. A hackathon demo that throws is a disqualified demo.
- **Marimo work is deferred.** Do not start until the ML endpoint is live.
- **All Databricks calls go through `app/api/predict/route.ts`** — browser code calls that route, never the endpoint directly.

## REST API Contract (Dashboard → Databricks Endpoint)

POST https://<workspace>.azuredatabricks.net/serving-endpoints/gulf-watch-v1/invocations

**Request:**
{
  "date": "YYYY-MM-DD",          // date to generate heatmap for
  "sst_delta": float,            // SST warming offset in °C (0 = baseline, maps to +0.025m SSH per °C)
  "loop_depth": float            // Loop Current penetration 0–1 (reserved for Marimo, pass 0.5 as default)
}

**Response:**
{
  "ri_probability": float[][],        // 2D grid, 0-1 scale, Gulf bounding box
  "lce_separation_prob_7d": float,    // model output: separation probability at t+7
  "lce_separation_prob_30d": float,   // model output: separation probability at t+30
  "ri_days_per_year": float,          // count of grid cells above 0.17m threshold, annualized
  "highest_risk_zone": string         // human-readable label e.g. "Eastern Gulf, ~150mi south of Tampa"
}

Endpoint URL must be read from environment variable GULF_WATCH_ENDPOINT — no hardcoded credentials.
Stub response format for local UI dev while model is training:
{
  "ri_probability": [[0.0]*20]*20,
  "lce_separation_prob_7d": 0.5,
  "lce_separation_prob_30d": 0.5,
  "ri_days_per_year": 0.0,
  "highest_risk_zone": "stub — model not connected"
}
## Model Evaluation Metrics

**Primary metric:** AUC-ROC
- Handles class imbalance (LCE separation events are rare relative to 40-year record)
- Target: AUC-ROC > 0.70 on validation set

**Secondary metric:** F1 score
- Use threshold 0.5 on sigmoid output for binary classification
- Report precision and recall separately for interpretability

**Do not use MAE or RMSE** — these are regression metrics. 
The model output is a probability (0–1), not a continuous value.

**Katrina sanity check (required before demo):**
- Plot lce_separation_prob_7d over the 90 days preceding Aug 28, 2005
- Expected: probability rises 2–4 weeks before Aug 28
- Print t+7 and t+30 probabilities for Aug 25, 2005 specifically
- If probability is flat or near-zero on this date, model has not learned the signal — 
  check label generation and class balance before proceeding
## Key science constants

- **SSH > 17 cm = RI threshold** (Mainelli et al. 2008). Colormap should inflect here.
- Training window: **1984–2020**. Validation window: **2020–2024**.
- Validation storms: **Katrina, Rita, Harvey, Ida, Michael**.
- Demo-hook target: **+2.4 °C SST → +9.6 RI events/year**. The stub honors this.

## Dev workflow

```bash
cp .env.example .env.local    # fill in Databricks URL + token when available
npm install
npm run dev                    # http://localhost:3000
npm test                       # client + stub tests
```

With `.env.local` unset the app runs entirely on stub data — safe for UI work.

## Layout

- `app/` — Next.js App Router pages + API routes.
- `app/api/predict/route.ts` — server proxy to Databricks.
- `lib/databricks/` — typed client, zod schemas, deterministic stub.
- `lib/databricks/__tests__/` — fallback + schema tests.
