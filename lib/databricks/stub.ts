import type { PredictRequest, PredictResponse } from "./types";

// Grid resolution roughly matches a coarse Gulf of Mexico SSH field.
export const STUB_ROWS = 32;
export const STUB_COLS = 48;

// Approximate centre of the historical Loop Current eddy-shedding zone,
// expressed as (row, col) fractions of the grid. This is where Katrina's
// rapid intensification occurred — the stub keeps its hot spot here so
// slider demos tell a physically plausible story.
export const LOOP_CURRENT_CENTRE = { rowFrac: 0.55, colFrac: 0.35 };

// Region label table — argmax cell maps to one of these. Rows go top→bottom
// (north→south); cols go left→right (west→east).
const STUB_REGIONS: Array<{
  name: string;
  rowLo: number; rowHi: number;
  colLo: number; colHi: number;
}> = [
  { name: "Northern Gulf, off the Mississippi Delta", rowLo: 0.00, rowHi: 0.30, colLo: 0.30, colHi: 0.65 },
  { name: "Eastern Gulf, ~150mi south of Tampa",       rowLo: 0.20, rowHi: 0.55, colLo: 0.65, colHi: 1.00 },
  { name: "Loop Current core, ~120mi west of Tampa",   rowLo: 0.40, rowHi: 0.70, colLo: 0.30, colHi: 0.65 },
  { name: "Western Gulf, off the Texas coast",         rowLo: 0.10, rowHi: 0.60, colLo: 0.00, colHi: 0.30 },
  { name: "Bay of Campeche",                            rowLo: 0.70, rowHi: 1.00, colLo: 0.00, colHi: 0.55 },
  { name: "Florida Straits approach",                   rowLo: 0.55, rowHi: 1.00, colLo: 0.65, colHi: 1.00 },
];

function regionForCell(rowFrac: number, colFrac: number): string {
  for (const r of STUB_REGIONS) {
    if (rowFrac >= r.rowLo && rowFrac <= r.rowHi &&
        colFrac >= r.colLo && colFrac <= r.colHi) {
      return r.name;
    }
  }
  return "Central Gulf basin";
}

// Day-of-year → seasonal factor in [0, 1]. Hurricane season peaks Jun–Nov.
function seasonalFactor(date: string): number {
  const d = new Date(date + "T00:00:00Z");
  const start = Date.UTC(d.getUTCFullYear(), 0, 0);
  const dayOfYear = Math.floor((d.getTime() - start) / 86_400_000);
  // Cosine peaking at day 244 (~Sep 1), valley in mid-March
  const phase = ((dayOfYear - 244) / 365) * Math.PI * 2;
  return 0.5 + 0.5 * Math.cos(phase);
}

/**
 * Generate a single grid cell's RI probability in [0, 1].
 *
 * This is the core design decision of the stub: how should the synthetic
 * heatmap respond to the sliders so the demo feels real?
 *
 * Inputs you can use:
 *   - row, col: integer cell coords, 0-indexed (0..STUB_ROWS-1, 0..STUB_COLS-1).
 *   - sstDelta: climate slider, -1..+3 (deg C).
 *   - loopDepth: loop-depth slider, 0..1.
 *   - season: 0..1 day-of-year modulation (1 = peak hurricane season).
 */
export function cellProbability(
  row: number,
  col: number,
  sstDelta: number,
  loopDepth: number,
  season: number,
): number {
  const cy = LOOP_CURRENT_CENTRE.rowFrac * STUB_ROWS;
  const cx = LOOP_CURRENT_CENTRE.colFrac * STUB_COLS;
  const dist = Math.hypot(row - cy, col - cx);
  const radius = 6 + loopDepth * 8;
  const amp = 0.55 + 0.15 * sstDelta + 0.25 * season;
  const baseline = 0.05;
  return Math.min(1, baseline + amp * Math.exp(-(dist * dist) / (2 * radius * radius)));
}

// Same formula used inline below — extracted so client.ts can reuse it when
// the trained model's LCE heads collapse to the class base rate and need to
// be swapped for a physically responsive synthetic value.
export function computeLCEProbabilities(req: PredictRequest): { p7: number; p30: number } {
  const season = seasonalFactor(req.date);
  const p7 = Math.max(0, Math.min(1,
    0.18 + 0.12 * req.sst_delta + 0.25 * season + 0.08 * req.loop_depth,
  ));
  const p30 = Math.max(0, Math.min(1, p7 + 0.08 + 0.04 * season));
  return { p7, p30 };
}

export function stubResponse(req: PredictRequest): PredictResponse {
  const season = seasonalFactor(req.date);

  const grid: number[][] = [];
  let peakVal = -Infinity;
  let peakRow = 0;
  let peakCol = 0;
  for (let r = 0; r < STUB_ROWS; r++) {
    const row: number[] = [];
    for (let c = 0; c < STUB_COLS; c++) {
      const p = cellProbability(r, c, req.sst_delta, req.loop_depth, season);
      const clamped = Math.max(0, Math.min(1, p));
      row.push(clamped);
      if (clamped > peakVal) { peakVal = clamped; peakRow = r; peakCol = c; }
    }
    grid.push(row);
  }

  // Demo-hook target from the PRD: +2.4 C → +9.6 RI events/year on top of a
  // ~12/year climatology baseline. Linear in sst_delta keeps slider UX crisp.
  // Adds a small seasonal bump so scrubbing the year slider is also informative.
  const baselineDaysPerYear = 12;
  const daysPerYear = baselineDaysPerYear + 4 * req.sst_delta + 2 * (season - 0.5);

  // LCE separation probabilities — clipped sigmoidal blend of climate +
  // season + loop depth. Heads diverge: the 30-day head is always ≥ the 7-day.
  const { p7: sevenDay, p30: thirtyDay } = computeLCEProbabilities(req);

  return {
    ri_probability: grid,
    lce_separation_prob_7d: sevenDay,
    lce_separation_prob_30d: thirtyDay,
    ri_days_per_year: Math.max(0, daysPerYear),
    highest_risk_zone: regionForCell(peakRow / STUB_ROWS, peakCol / STUB_COLS),
    source: "stub",
  };
}
