import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PredictResponseSchema, type PredictRequest } from "../types";

const baseReq: PredictRequest = {
  date: "2005-08-25",
  sst_delta: 0,
  loop_depth: 0.5,
};

async function freshClient() {
  vi.resetModules();
  return (await import("../client")).predictRI;
}

describe("predictRI", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    vi.restoreAllMocks();
    delete process.env.DATABRICKS_ENDPOINT_URL;
    delete process.env.DATABRICKS_TOKEN;
  });

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  it("returns a stub when env vars are unset", async () => {
    const predictRI = await freshClient();
    const res = await predictRI(baseReq);
    expect(res.source).toBe("stub");
    expect(PredictResponseSchema.safeParse(res).success).toBe(true);
  });

  it("returns a stub when fetch rejects", async () => {
    process.env.DATABRICKS_ENDPOINT_URL = "https://example.invalid/endpoint";
    process.env.DATABRICKS_TOKEN = "test-token";
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));
    const predictRI = await freshClient();
    const res = await predictRI(baseReq);
    expect(res.source).toBe("stub");
  });

  it("returns a stub when the response fails schema validation", async () => {
    process.env.DATABRICKS_ENDPOINT_URL = "https://example.invalid/endpoint";
    process.env.DATABRICKS_TOKEN = "test-token";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ wrong: "shape" }),
      }),
    );
    const predictRI = await freshClient();
    const res = await predictRI(baseReq);
    expect(res.source).toBe("stub");
  });

  it("passes through a valid Databricks response", async () => {
    process.env.DATABRICKS_ENDPOINT_URL = "https://example.invalid/endpoint";
    process.env.DATABRICKS_TOKEN = "test-token";
    const good = {
      ri_probability: [[0.1, 0.2], [0.3, 0.4]],
      lce_separation_prob_7d: 0.62,
      lce_separation_prob_30d: 0.78,
      ri_days_per_year: 12.5,
      highest_risk_zone: "Eastern Gulf, ~150mi south of Tampa",
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => good }),
    );
    const predictRI = await freshClient();
    const res = await predictRI(baseReq);
    expect(res.source).toBe("databricks");
    expect(res.ri_days_per_year).toBe(12.5);
    expect(res.lce_separation_prob_7d).toBe(0.62);
    expect(res.highest_risk_zone).toBe("Eastern Gulf, ~150mi south of Tampa");
  });

  it("different sst_delta values produce different stub outputs", async () => {
    const predictRI = await freshClient();
    const low = await predictRI({ ...baseReq, sst_delta: 0 });
    const high = await predictRI({ ...baseReq, sst_delta: 2.4 });
    // ri_days_per_year varies linearly with sst_delta via stubResponse.
    expect(high.ri_days_per_year).toBeGreaterThan(low.ri_days_per_year);
    // lce_separation_prob_7d also responds to climate change.
    expect(high.lce_separation_prob_7d).toBeGreaterThan(low.lce_separation_prob_7d);
  });
});
