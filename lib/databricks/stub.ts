import type { PredictRequest, PredictResponse } from "./types";

// Grid resolution roughly matches a coarse Gulf of Mexico SSH field.
export const STUB_ROWS = 32;
export const STUB_COLS = 48;

// Approximate centre of the historical Loop Current eddy-shedding zone,
// expressed as (row, col) fractions of the grid. This is where Katrina's
// rapid intensification occurred — the stub should keep its hot spot here
// so slider demos tell a physically plausible story.
export const LOOP_CURRENT_CENTRE = { rowFrac: 0.55, colFrac: 0.35 };

/**
 * Generate a single grid cell's RI probability in [0, 1].
 *
 * This is the core design decision of the stub: how should the synthetic
 * heatmap respond to the sliders so the demo feels real?
 *
 * Inputs you can use:
 *   - row, col: integer cell coords, 0-indexed.
 *   - sstDelta: climate slider, -1..+3 (deg C).
 *   - loopDepth: loop-depth slider, 0..1.
 *
 * See the user contribution block in stub.ts for guidance.
 */
export function cellProbability(
  row: number,
  col: number,
  sstDelta: number,
  loopDepth: number,
): number {
  // --- USER CONTRIBUTION ---
  // TODO(user): implement the RI probability field. See the explanation
  // from the assistant for trade-offs. Must return a value in [0, 1].
  // Suggested shape: a Gaussian hot spot at LOOP_CURRENT_CENTRE whose
  // amplitude grows with sstDelta and whose radius grows with loopDepth,
  // plus a small baseline so the rest of the Gulf isn't pure zero.
  return 0;
  // --- END USER CONTRIBUTION ---
}

export function stubResponse(req: PredictRequest): PredictResponse {
  const grid: number[][] = [];
  for (let r = 0; r < STUB_ROWS; r++) {
    const row: number[] = [];
    for (let c = 0; c < STUB_COLS; c++) {
      const p = cellProbability(r, c, req.sst_delta, req.loop_depth);
      row.push(Math.max(0, Math.min(1, p)));
    }
    grid.push(row);
  }

  // Demo-hook target from the PRD: +2.4 C -> +9.6 RI events/year on top of a
  // ~12/year climatology baseline. Linear in sst_delta keeps slider UX crisp.
  const baselineDaysPerYear = 12;
  const daysPerYear = baselineDaysPerYear + 4 * req.sst_delta;

  return {
    ri_probability: grid,
    ri_days_per_year: Math.max(0, daysPerYear),
    mae: 0.08,
    source: "stub",
  };
}
