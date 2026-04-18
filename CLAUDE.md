# Gulf Watch

Hackathon project (Climate + AI/ML track, due **2026-04-18**). Compresses 40 years of Gulf of Mexico sea surface height (SSH) data into a real-time hurricane rapid-intensification (RI) prediction dashboard. Demo hook: drag an SST slider, watch the RI heatmap intensify and the events-per-year counter climb.

Full PRD: `~/Downloads/gulf_watch_general_prd.html`.

## Three-layer architecture

1. **Databricks (brain)** — ConvLSTM on raw SSH `.nc` files, MLflow-tracked, served as a REST endpoint. *Status: in progress, separate workstream.*
2. **Marimo notebook (science UI)** — reactive notebook for Scripps judges. *Status: deferred until Databricks endpoint is live.*
3. **Dashboard UI (judge experience)** — Next.js + TypeScript + deck.gl. 5 panels: 40-year timelapse, hurricane scrubber, prediction heatmap, climate sliders, ElevenLabs voice agent. *Status: mockup done elsewhere; wire to API once ported in.*

## Hard rules

- **Do NOT redesign the dashboard UI.** The mockup is the source of truth — wire it, don't rebuild it.
- **Databricks endpoint URL comes from `DATABRICKS_ENDPOINT_URL`.** Never hardcode. Token comes from `DATABRICKS_TOKEN`. Both are server-side only; never expose to the browser.
- **Stub fallback is mandatory.** If env vars are missing, the request fails, or the response fails schema validation, fall back to the stub. A hackathon demo that throws is a disqualified demo.
- **Marimo work is deferred.** Do not start until the ML endpoint is live.
- **All Databricks calls go through `app/api/predict/route.ts`** — browser code calls that route, never the endpoint directly.

## API contract

`POST $DATABRICKS_ENDPOINT_URL` (proxied via `/api/predict`):

```
Request:  { ssh_window: float[][], sst_delta: float, loop_depth: float }
Response: { ri_probability: float[][], ri_days_per_year: float, mae: float }
```

Types + zod schemas live in `lib/databricks/types.ts`. The client tags stub responses with `source: "stub"` so the UI can show a "stub data" indicator during dev.

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
