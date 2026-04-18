# Gulf Watch

Hurricane rapid-intensification prediction dashboard. Next.js + TypeScript, backed by a Databricks ConvLSTM served as REST.

See [CLAUDE.md](./CLAUDE.md) for architecture, hard rules, and the API contract.

## Quickstart

```bash
cp .env.example .env.local   # optional — app runs on stub data if unset
npm install
npm run dev                  # http://localhost:3000
npm test                     # client + stub fallback tests
```

With no `.env.local`, every `/api/predict` call returns deterministic stub data tagged `source: "stub"`. This is the intended dev mode while the ML endpoint is still training.

## Layout

```
app/
  api/predict/route.ts   # server proxy to Databricks
  page.tsx               # placeholder — replace with dashboard mockup
lib/databricks/
  client.ts              # predictRI(): env-aware, timeout, stub fallback
  stub.ts                # deterministic synthetic heatmap
  types.ts               # zod schemas + types
```
