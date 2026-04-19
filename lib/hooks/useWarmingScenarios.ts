"use client";

import { useEffect, useRef, useState } from "react";
import type { PredictResponse } from "@/lib/databricks/types";
import { tToDate } from "./usePrediction";

const SCENARIO_DELTAS = [0, 1, 2, 3, 4] as const;
const DEBOUNCE_MS = 300;

export interface WarmingScenario {
  sst_delta: number;
  data: PredictResponse | null;
}

interface UseWarmingScenariosResult {
  scenarios: WarmingScenario[];
  loading: boolean;
}

/**
 * Fetches the predict endpoint at four warming levels (+0, +1, +2, +3 °C) so
 * the sidebar can show the *actual* model sensitivity curve instead of a
 * linear extrapolation. Keyed on date + loopDepth — anomaly slider drag
 * does NOT refire these because the scenarios already span the full range.
 */
export function useWarmingScenarios(t: number, loopDepth: number): UseWarmingScenariosResult {
  const [scenarios, setScenarios] = useState<WarmingScenario[]>(
    SCENARIO_DELTAS.map(d => ({ sst_delta: d, data: null })),
  );
  const [loading, setLoading] = useState(false);
  const reqIdRef = useRef(0);

  useEffect(() => {
    const date = tToDate(t);
    const myId = ++reqIdRef.current;
    setLoading(true);

    const handle = setTimeout(async () => {
      try {
        const results = await Promise.all(
          SCENARIO_DELTAS.map(async d => {
            const res = await fetch("/api/predict", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ date, sst_delta: d, loop_depth: loopDepth }),
            });
            if (!res.ok) return { sst_delta: d, data: null };
            const json: PredictResponse = await res.json();
            return { sst_delta: d, data: json };
          }),
        );
        if (myId !== reqIdRef.current) return; // stale
        setScenarios(results);
        setLoading(false);
      } catch {
        if (myId !== reqIdRef.current) return;
        setLoading(false);
      }
    }, DEBOUNCE_MS);

    return () => clearTimeout(handle);
  }, [t, loopDepth]);

  return { scenarios, loading };
}
