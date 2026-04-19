"use client";

import { useEffect, useRef, useState } from "react";
import type { PredictResponse } from "@/lib/databricks/types";

const DEBOUNCE_MS = 200;

/**
 * Convert the dashboard's normalised time slider t ∈ [0, 1] into a calendar
 * date string the API expects. The 40-year window starts 1985-01-01.
 *
 * Lives here (not in dashboard.tsx) so the hook owns the wire-format mapping
 * and components stay UI-only.
 */
export function tToDate(t: number): string {
  const start = Date.UTC(1985, 0, 1);
  const span = 40 * 365 * 86_400_000;
  const ms = start + Math.max(0, Math.min(1, t)) * span;
  return new Date(ms).toISOString().slice(0, 10);
}

interface UsePredictionResult {
  data: PredictResponse | null;
  loading: boolean;
  error: string | null;
  source: PredictResponse["source"] | null;
}

/**
 * Calls /api/predict whenever (t, sstDelta, loopDepth) change. Debounced so
 * dragging a slider doesn't fire 60 reqs/sec. Keeps the previous response
 * visible while a new request is in flight (no flicker on slider drag).
 *
 * Out-of-order responses are discarded via a monotonically-incrementing
 * request id so the latest user input always wins.
 */
export function usePrediction(
  t: number,
  sstDelta: number,
  loopDepth: number,
): UsePredictionResult {
  const [data, setData] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const reqIdRef = useRef(0);

  useEffect(() => {
    const date = tToDate(t);
    const myId = ++reqIdRef.current;
    setLoading(true);

    const handle = setTimeout(async () => {
      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ date, sst_delta: sstDelta, loop_depth: loopDepth }),
        });
        if (myId !== reqIdRef.current) return; // stale
        if (!res.ok) {
          setError(`HTTP ${res.status}`);
          setLoading(false);
          return;
        }
        const json: PredictResponse = await res.json();
        if (myId !== reqIdRef.current) return; // stale
        setData(json);
        setError(null);
        setLoading(false);
      } catch (e) {
        if (myId !== reqIdRef.current) return;
        setError(e instanceof Error ? e.message : "request failed");
        setLoading(false);
      }
    }, DEBOUNCE_MS);

    return () => clearTimeout(handle);
  }, [t, sstDelta, loopDepth]);

  return { data, loading, error, source: data?.source ?? null };
}
