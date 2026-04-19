import { z } from "zod";

// Request shape the dashboard sends. `date` is the ISO calendar date the user
// has scrubbed to (the model slices its own 30-day window ending on this
// date from the SSH NetCDF on Databricks). `sst_delta` is the climate-slider
// offset in degrees C (-1..+3, baseline 0). `loop_depth` is a normalised
// 0..1 Loop Current depth (currently reserved for the Marimo tier — the
// served model accepts it but no-ops it).
//
// This schema matches the REST contract documented in CLAUDE.md exactly.
export const PredictRequestSchema = z.object({
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  sst_delta: z.number().min(-1).max(3),
  loop_depth: z.number().min(0).max(1),
});
export type PredictRequest = z.infer<typeof PredictRequestSchema>;

// Databricks response. ri_probability is a 2D risk grid for the Gulf bbox,
// values in [0, 1]. lce_separation_prob_{7,30}d are the LSTM's two sigmoid
// heads (probability of an LCE separation event within the next 7 / 30 days).
// ri_days_per_year is an annualised count derived from the 365-day window
// centred on `date`. highest_risk_zone is a human-readable label of the
// argmax cell.
export const PredictResponseSchema = z.object({
  ri_probability: z.array(z.array(z.number().min(0).max(1))).min(1),
  lce_separation_prob_7d: z.number().min(0).max(1),
  lce_separation_prob_30d: z.number().min(0).max(1),
  ri_days_per_year: z.number().min(0),
  highest_risk_zone: z.string(),
});
export type PredictResponseBody = z.infer<typeof PredictResponseSchema>;

// Internal type — we tag every response so the UI can show a "stub" badge
// during dev without a second API call.
export type PredictResponse = PredictResponseBody & {
  source: "databricks" | "stub";
};
