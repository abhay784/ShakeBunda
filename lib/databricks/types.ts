import { z } from "zod";

// Request shape the dashboard sends. ssh_window is a 2D spatial grid of SSH
// values (metres) for a rolling 30-day window. sst_delta is the climate-slider
// offset in degrees C. loop_depth is a normalised 0..1 depth for the Loop
// Current (shallow..deep).
export const PredictRequestSchema = z.object({
  ssh_window: z.array(z.array(z.number())).min(1),
  sst_delta: z.number().min(-1).max(3),
  loop_depth: z.number().min(0).max(1),
});
export type PredictRequest = z.infer<typeof PredictRequestSchema>;

// Databricks response. ri_probability is a 2D grid aligned with the input
// window, values in [0, 1]. ri_days_per_year is a scalar annualised count.
// mae is the model's validation MAE (carried through for the UI badge).
export const PredictResponseSchema = z.object({
  ri_probability: z.array(z.array(z.number().min(0).max(1))).min(1),
  ri_days_per_year: z.number().min(0),
  mae: z.number().min(0),
});
export type PredictResponseBody = z.infer<typeof PredictResponseSchema>;

// Internal type — we tag every response so the UI can show a "stub" badge
// during dev without a second API call.
export type PredictResponse = PredictResponseBody & {
  source: "databricks" | "stub";
};
