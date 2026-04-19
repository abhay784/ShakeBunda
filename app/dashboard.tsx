'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { GulfMap } from '@/components/GulfMap';
import { LeftPanel, RightPanel } from '@/components/GulfPanels';
import { predictRI } from '@/lib/databricks/client';
import type { Eddy } from '@/lib/ocean/ssh';
import type { PredictResponse } from '@/lib/databricks/types';
import './globals.css';

interface RiskRegion {
  name: string;
  x: number;
  y: number;
  prob: number;
  ssh: string;
  trend: string;
}

export default function Dashboard() {
  const [t, setT] = useState<number>(() => {
    if (typeof window === 'undefined') return 0;
    const saved = parseFloat(localStorage.getItem('gw.t') || '');
    return isNaN(saved) ? (2021 - 1985) / 40 : saved;
  });

  const [anomaly, setAnomaly] = useState(0.9);
  const [layers, setLayers] = useState({
    eddies: true,
    loopCurrent: true,
    riskZones: true,
    predictions: false,
  });
  const [mode, setMode] = useState<'historical' | 'predicted'>('historical');
  const [speed, setSpeed] = useState(5);
  const [playing, setPlaying] = useState(false);
  const [eddyCount, setEddyCount] = useState(0);
  const [showGrid, setShowGrid] = useState(true);
  const [showScanlines, setShowScanlines] = useState(true);
  const [density, setDensity] = useState(1.0);
  const [glow, setGlow] = useState(1.0);
  const [riGrid, setRiGrid] = useState<number[][] | undefined>();
  const [loading, setLoading] = useState(false);
  const [eddies, setEddies] = useState<Eddy[]>([]);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);

  // Persist time
  useEffect(() => {
    localStorage.setItem('gw.t', t.toString());
  }, [t]);

  // Playback
  useEffect(() => {
    if (!playing) return;
    let raf: number;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      setT((prev) => {
        const nt = prev + (dt / 40) * (speed / 1);
        return nt > 1 ? 0 : nt;
      });
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed]);

  // Fetch RI predictions. We always fetch (both modes) so the Ocean Risk
  // Engine gauge has live data; only the heatmap override is mode-gated.
  useEffect(() => {
    let cancelled = false;
    const fetchPredictions = async () => {
      setLoading(true);
      try {
        const result = await predictRI({
          ssh_window: [[anomaly]],
          sst_delta: anomaly,
          loop_depth: 0.5,
        });
        if (cancelled) return;
        setPrediction(result);
        if (mode === 'predicted' && result.ri_probability?.length > 0) {
          setRiGrid(result.ri_probability);
        } else {
          setRiGrid(undefined);
        }
      } catch (err) {
        console.error('Failed to fetch RI predictions:', err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchPredictions();
    return () => { cancelled = true; };
  }, [mode, anomaly]);

  const accentHex = '#4dd6ff';

  const handleFlyTo = useCallback((r: RiskRegion) => {
    // Placeholder for future region pin animation
    console.log('Flying to region:', r.name);
  }, []);

  return (
    <div
      className={`gw-app ${showScanlines ? 'scan' : ''}`}
      style={{ '--accent': accentHex } as React.CSSProperties}
    >
      <LeftPanel
        t={t}
        setT={setT}
        anomaly={anomaly}
        setAnomaly={setAnomaly}
        layers={layers}
        setLayers={setLayers}
        mode={mode}
        setMode={setMode}
        speed={speed}
        setSpeed={setSpeed}
        playing={playing}
        setPlaying={setPlaying}
      />

      <main className="gw-main">
        <div className="gw-topbar">
          <div className="gw-top-left">
            <span className="gw-breadcrumb active">GULF · 18–31°N · 98–80°W</span>
            <span className="gw-top-chip">
              <i className="gw-dot live" />
              {mode === 'predicted' ? 'ORM-v3' : 'HINDCAST'}
            </span>
          </div>
          <div className="gw-top-right">
            <span className="gw-top-stat">{eddyCount} eddies</span>
          </div>
        </div>

        <div className="gw-stage">
          <GulfMap
            t={t}
            anomaly={anomaly}
            layers={layers}
            density={density}
            glow={glow}
            showGrid={showGrid}
            onEddyCount={setEddyCount}
            onEddiesChange={setEddies}
            mode={mode}
            riGrid={riGrid}
          />

          <div className="gw-stage-chrome">
            <div className="gw-legend">
              <div className="gw-legend-h">SSH ANOMALY · cm</div>
              <div className="gw-legend-bar">
                <span className="gw-legend-grad" />
                <span className="gw-legend-ticks">
                  <i>−20</i>
                  <i>−10</i>
                  <i>0</i>
                  <i>+10</i>
                  <i>+17</i>
                  <i>+25</i>
                </span>
              </div>
              <div className="gw-legend-note">
                <span className="gw-pip" /> SSH &gt; +17 cm ·{' '}
                <b>rapid intensification potential</b>
              </div>
            </div>
          </div>
        </div>

        <div className="gw-bottom">
          <div className="gw-analytics" />
        </div>
      </main>

      <RightPanel
        t={t}
        anomaly={anomaly}
        eddies={eddies}
        prediction={prediction}
        mode={mode}
        onFly={handleFlyTo}
      />
    </div>
  );
}
