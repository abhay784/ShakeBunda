import {
  PredictResponseSchema,
  type PredictRequest,
  type PredictResponse,
} from "./types";
import { stubResponse } from "./stub";

const REQUEST_TIMEOUT_MS = 8_000;

let missingEnvWarned = false;
function warnMissingEnv(reason: string) {
  if (missingEnvWarned) return;
  missingEnvWarned = true;
  // eslint-disable-next-line no-console
  console.warn(`[databricks] ${reason} — using stub responses.`);
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

    return { ...parsed.data, source: "databricks" };
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn("[databricks] fetch failed — falling back to stub.", err);
    return stubResponse(req);
  } finally {
    clearTimeout(timer);
  }
}
