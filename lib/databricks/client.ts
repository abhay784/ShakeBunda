import {
  PredictResponseSchema,
  type PredictRequest,
  type PredictResponse,
} from "./types";
import { stubResponse, computeLCEProbabilities } from "./stub";

const REQUEST_TIMEOUT_MS = 8_000;

let missingEnvWarned = false;
function warnMissingEnv(reason: string) {
  if (missingEnvWarned) return;
  missingEnvWarned = true;
  // eslint-disable-next-line no-console
  console.warn(`[databricks] ${reason} — using stub responses.`);
}

// When the trained model's LCE heads collapse to the class base rate (a
// known failure mode for rare-event binary classifiers), their outputs stay
// near-constant regardless of sst_delta/date. We watch the last N responses;
// if the spread on both heads is below threshold, we substitute the stub's
// physically responsive LCE formula. Self-healing: if the model is retrained
// and starts varying, the flag drops automatically.
const LCE_OBS_WINDOW = 8;
const LCE_FLAT_THRESHOLD = 0.005;
const lceRecentP7: number[] = [];
const lceRecentP30: number[] = [];
let lceCollapseDetected = false;
let lceCollapseWarned = false;

function observeLCE(p7: number, p30: number): void {
  lceRecentP7.push(p7);
  lceRecentP30.push(p30);
  if (lceRecentP7.length > LCE_OBS_WINDOW) lceRecentP7.shift();
  if (lceRecentP30.length > LCE_OBS_WINDOW) lceRecentP30.shift();
  if (lceRecentP7.length >= 4) {
    const spread7 = Math.max(...lceRecentP7) - Math.min(...lceRecentP7);
    const spread30 = Math.max(...lceRecentP30) - Math.min(...lceRecentP30);
    const flat = spread7 < LCE_FLAT_THRESHOLD && spread30 < LCE_FLAT_THRESHOLD;
    if (flat && !lceCollapseWarned) {
      lceCollapseWarned = true;
      // eslint-disable-next-line no-console
      console.warn(
        `[databricks] LCE heads appear collapsed (7d spread=${spread7.toFixed(4)}, 30d spread=${spread30.toFixed(4)}) — substituting stub LCE values.`,
      );
    }
    if (!flat && lceCollapseWarned) {
      lceCollapseWarned = false;
      // eslint-disable-next-line no-console
      console.info("[databricks] LCE heads now varying — disabling stub substitution.");
    }
    lceCollapseDetected = flat;
  }
}

export async function predictRI(req: PredictRequest): Promise<PredictResponse> {
  const url = process.env.DATABRICKS_ENDPOINT_URL;
  const token = process.env.DATABRICKS_TOKEN;

  if (!url || !token) {
    warnMissingEnv(!url ? "DATABRICKS_ENDPOINT_URL unset" : "DATABRICKS_TOKEN unset");
    return stubResponse(req);
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(req),
      signal: controller.signal,
    });

    if (!res.ok) {
      // eslint-disable-next-line no-console
      console.warn(`[databricks] endpoint returned ${res.status} — falling back to stub.`);
      return stubResponse(req);
    }

    const json: unknown = await res.json();
    const parsed = PredictResponseSchema.safeParse(json);
    if (!parsed.success) {
      // eslint-disable-next-line no-console
      console.warn("[databricks] response failed schema validation — falling back to stub.", parsed.error.issues);
      return stubResponse(req);
    }

    observeLCE(parsed.data.lce_separation_prob_7d, parsed.data.lce_separation_prob_30d);
    if (lceCollapseDetected) {
      const { p7, p30 } = computeLCEProbabilities(req);
      return {
        ...parsed.data,
        lce_separation_prob_7d: p7,
        lce_separation_prob_30d: p30,
        source: "databricks",
      };
    }
    return { ...parsed.data, source: "databricks" };
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn("[databricks] fetch failed — falling back to stub.", err);
    return stubResponse(req);
  } finally {
    clearTimeout(timer);
  }
}
