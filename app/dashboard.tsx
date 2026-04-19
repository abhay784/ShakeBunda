'use client';

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { GulfMap } from '@/components/GulfMap';
import { LeftPanel, RightPanel, Analytics, WarmingHero, regionName } from '@/components/GulfPanels';
import { Timeline } from '@/components/GulfTimeline';
import { buildGrid, detectEddies } from '@/lib/ocean/ssh';
import { usePrediction } from '@/lib/hooks/usePrediction';
import { useWarmingScenarios } from '@/lib/hooks/useWarmingScenarios';

// Future-mode timeline: 2025 (today) → 2100 (IPCC end-of-century horizon).
const FUTURE_START = 2025;
const FUTURE_END = 2100;
const FUTURE_SPAN = FUTURE_END - FUTURE_START;

export function futureYearFromT(t: number): number {
  return FUTURE_START + Math.max(0, Math.min(1, t)) * FUTURE_SPAN;
}

// Accelerated warming curve — SSP5-8.5 "high emissions / business as usual".
// Power-law n=1.4 matches the convex shape of CMIP6 Gulf SST projections
// (Wang 2023, Karnauskas 2023, Muhling 2018). 2025 → 0°C, 2100 → +4.0°C.
// Cross-checks: 2040 ≈ +0.4, 2060 ≈ +1.4, 2080 ≈ +2.6, 2100 = +4.0°C —
// within the IPCC AR6 "likely" range for SSP5-8.5 Gulf regional warming.
export const WARMING_MAX = 4;
export function warmingCurve(year: number): number {
  const clamped = Math.max(FUTURE_START, Math.min(FUTURE_END, year));
  const frac = (clamped - FUTURE_START) / FUTURE_SPAN;
  return Math.pow(frac, 1.4) * WARMING_MAX;
}

function genSeries() {
  const ts: number[] = [], es: number[] = [], ps: number[] = [];
  const N = 120;
  for (let i = 0; i < N; i++) {
    const tt = i / (N - 1);
    const g = buildGrid(tt, 0.8);
    let sum = 0, cnt = 0;
    for (let k = 0; k < g.length; k++) {
      if (!isNaN(g[k])) { sum += g[k]; cnt++; }
    }
    ts.push((sum / cnt) * 17);
    es.push(detectEddies(g).length);
    ps.push((sum / cnt) * 17 + (Math.sin(tt * 30) * 0.6 + (Math.random() - 0.5) * 0.4));
  }
  return { ts, es, ps };
}

interface Pin {
  name: string;
  x: number;
  y: number;
  prob: number;
}

export default function Dashboard() {
  const [t, setT] = useState<number>(() => {
    if (typeof window === 'undefined') return (2021 - 1985) / 40;
    const saved = parseFloat(localStorage.getItem('gw.t') || '');
    return isNaN(saved) ? (2021 - 1985) / 40 : saved;
  });
  const [anomaly, setAnomaly] = useState(0.9);
  const [loopDepth, setLoopDepth] = useState<number>(() => {
    if (typeof window === 'undefined') return 0.5;
    const saved = parseFloat(localStorage.getItem('gw.loopDepth') || '');
    return isNaN(saved) ? 0.5 : saved;
  });
  const [layers, setLayers] = useState({
    eddies: true, loopCurrent: true, riskZones: true, predictions: false,
  });
  const [speed, setSpeed] = useState(5);
  const [playing, setPlaying] = useState(false);
  const [eddyCount, setEddyCount] = useState(0);
  const [tweakOpen, setTweakOpen] = useState(false);
  const [accent, setAccent] = useState('cyan');
  const [density, setDensity] = useState(1.0);
  const [glow, setGlow] = useState(1.0);
  const [showGrid, setShowGrid] = useState(true);
  const [showScanlines, setShowScanlines] = useState(true);
  const [pin, setPin] = useState<Pin | null>(null);
  const [series] = useState(() => genSeries());
  const [viewMode, setViewMode] = useState<'past' | 'future'>('past');
  // In future mode, anomaly auto-tracks the warming curve as time advances.
  // Manually dragging the temp slider sets this to false — user takes the wheel.
  const [anomalyLinked, setAnomalyLinked] = useState<boolean>(true);
  // Derived — keeps "mode" copy/labels consistent with the single view switch
  const mode = viewMode === 'future' ? 'predicted' : 'historical';

  // Persist time + loop depth
  useEffect(() => { localStorage.setItem('gw.t', t.toString()); }, [t]);
  useEffect(() => { localStorage.setItem('gw.loopDepth', loopDepth.toString()); }, [loopDepth]);

  // Re-link anomaly to the warming curve whenever we enter future mode.
  // Past mode leaves anomaly alone — it's a manual "what-if" slider there.
  useEffect(() => {
    if (viewMode === 'future') setAnomalyLinked(true);
  }, [viewMode]);

  // Future mode: drive anomaly from the warming curve as t advances (playback
  // OR manual scrub). A single reactive path handles both — beats threading
  // the ramp through the RAF loop.
  useEffect(() => {
    if (viewMode !== 'future' || !anomalyLinked) return;
    setAnomaly(warmingCurve(futureYearFromT(t)));
  }, [t, viewMode, anomalyLinked]);

  // Wrapped setter for the temp slider — manual drag breaks the auto-link so
  // user overrides aren't immediately stomped by the next ramp tick.
  const handleAnomalyChange = (v: number) => {
    if (viewMode === 'future' && anomalyLinked) setAnomalyLinked(false);
    setAnomaly(v);
  };

  const { data: prediction } = usePrediction(t, anomaly, loopDepth);
  const { scenarios: warmingScenarios, loading: scenariosLoading } = useWarmingScenarios(t, loopDepth);

  // Playback
  useEffect(() => {
    if (!playing) return;
    let raf: number;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = (now - last) / 1000; last = now;
      setT(prev => {
        const nt = prev + (dt / 40) * speed;
        return nt > 1 ? 0 : nt;
      });
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed]);

  // Tweaks postMessage listener
  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      if (!e.data) return;
      if (e.data.type === '__activate_edit_mode') setTweakOpen(true);
      if (e.data.type === '__deactivate_edit_mode') setTweakOpen(false);
    };
    window.addEventListener('message', onMsg);
    window.parent.postMessage({ type: '__edit_mode_available' }, '*');
    return () => window.removeEventListener('message', onMsg);
  }, []);

  function persistTweak(key: string, val: unknown) {
    window.parent.postMessage({ type: '__edit_mode_set_keys', edits: { [key]: val } }, '*');
  }

  function flyTo(r: Pin) {
    setPin(r);
    setTimeout(() => setPin(null), 2800);
  }

  const grid = useMemo(() => buildGrid(t, anomaly), [t, anomaly]);
  const eddies = useMemo(() => detectEddies(grid), [grid]);

  const accentHex = ({ cyan: '#4dd6ff', teal: '#00e0c4', amber: '#ffb347', violet: '#a078ff' } as Record<string, string>)[accent] ?? '#4dd6ff';

  const peakStrength = eddies.length > 0 ? Math.max(...eddies.map(e => e.strength * 17 + 8)) : 0;

  const baselinePct = useMemo(() => {
    const g0 = buildGrid(0, 0);
    const e0 = detectEddies(g0);
    if (e0.length === 0) return 0;
    const top = e0.reduce((a, b) => (a.strength > b.strength ? a : b));
    return Math.min(0.99, (top.strength - 0.35) * 1.15) * 100;
  }, []);

  const status = useMemo(() => {
    const histYear = 1985 + t * 40;
    const month = Math.floor((histYear - Math.floor(histYear)) * 12);
    const monthName = ['January','February','March','April','May','June','July','August','September','October','November','December'][month];
    const displayYear = viewMode === 'future' ? Math.floor(futureYearFromT(t)) : Math.floor(histYear);
    const n = eddies.length;
    const lc = n <= 1 ? 'intact' : (n >= 4 || peakStrength > 25 ? 'shedding' : 'extended');
    if (n === 0) {
      return `${monthName} ${displayYear} · Loop Current ${lc} · Central Gulf basin at low RI risk · 0 warm eddies detected`;
    }
    const top = eddies.reduce((a, b) => (a.strength > b.strength ? a : b));
    const prob = Math.min(0.99, (top.strength - 0.35) * 1.15 + anomaly * 0.08);
    const level = prob < 0.4 ? 'low' : prob < 0.7 ? 'elevated' : 'high';
    return `${monthName} ${displayYear} · Loop Current ${lc} · ${regionName(top.x, top.y)} at ${level} RI risk · ${n} warm eddies detected`;
  }, [t, anomaly, eddies, peakStrength, viewMode]);

  return (
    <div
      className={`gw-app ${showScanlines ? 'scan' : ''}`}
      style={{ '--accent': accentHex } as React.CSSProperties}
    >
      <LeftPanel
        t={t} setT={setT}
        anomaly={anomaly} setAnomaly={handleAnomalyChange}
        loopDepth={loopDepth} setLoopDepth={setLoopDepth}
        layers={layers} setLayers={setLayers}
        speed={speed} setSpeed={setSpeed}
        playing={playing} setPlaying={setPlaying}
        viewMode={viewMode} setViewMode={setViewMode}
      />

      <main className="gw-main">
        <div className="gw-topbar">
          <div className="gw-top-left">
            <span className="gw-breadcrumb active">GULF · 18–31°N · 98–80°W</span>
            <span className="gw-top-chip">
              <i className="gw-dot live" />{mode === 'predicted' ? 'ORM-v3' : 'Historical'}
            </span>
            {prediction?.source === 'stub' && (
              <span className="gw-top-chip gw-chip-demo">
                <i className="gw-dot" /> DEMO DATA
              </span>
            )}
            {prediction?.source === 'databricks' && (
              <span className="gw-top-chip gw-chip-live">
                <i className="gw-dot live" /> DATABRICKS
              </span>
            )}
          </div>
          <div className="gw-top-right">
            <span className="gw-top-stat">{eddyCount} eddies · {peakStrength.toFixed(1)}cm peak</span>
          </div>
        </div>

        <WarmingHero
          anomaly={anomaly}
          setAnomaly={handleAnomalyChange}
          primary={viewMode === 'future'}
          linked={viewMode === 'future' ? anomalyLinked : null}
          onRelink={() => setAnomalyLinked(true)}
          futureYear={viewMode === 'future' ? Math.floor(futureYearFromT(t)) : null}
        />

        <div className="gw-stage">
          <GulfMap
            t={t}
            anomaly={anomaly}
            layers={layers}
            density={density}
            glow={glow}
            showGrid={showGrid}
            onEddyCount={setEddyCount}
            mode={mode}
            riskOverlay={prediction?.ri_probability ?? null}
          />

          <div className="gw-stage-chrome">
            <div className="gw-crosshair">
              <div className="gw-crosshair-h" />
              <div className="gw-crosshair-v" />
              <div className="gw-crosshair-readout">
                <b>26.4°N · 88.2°W</b>
                <span>SSH <i>{(Math.sin(t * 30) * 8 + 12).toFixed(1)} cm</i></span>
                <span>SST <i>{(28.4 + anomaly).toFixed(1)} °C</i></span>
                <span>Ocean Heat Content <i>{(85 + anomaly * 18).toFixed(0)} kJ/cm²</i></span>
              </div>
            </div>

            {pin && (
              <div className="gw-pin" style={{ left: `${pin.x * 100}%`, top: `${(1 - pin.y) * 100}%` }}>
                <div className="gw-pin-ring" />
                <div className="gw-pin-ring d2" />
                <div className="gw-pin-lbl">{pin.name} · {(pin.prob * 100) | 0}%</div>
              </div>
            )}

            <div className="gw-stage-meta">
              <span>{status}</span>
            </div>
          </div>
        </div>

        <div className="gw-bottom">
          <Analytics tSeries={series.ts} eddySeries={series.es} tIdx={t} />
          <Timeline t={t} setT={setT} playing={playing} />
        </div>
      </main>

      <RightPanel
        t={t} anomaly={anomaly} grid={grid} eddies={eddies} mode={mode}
        onFly={flyTo}
        baselinePct={baselinePct}
        prediction={prediction}
        warmingScenarios={warmingScenarios}
        scenariosLoading={scenariosLoading}
      />

      {tweakOpen && (
        <div className="gw-tweaks">
          <div className="gw-tweaks-h">
            <b>TWEAKS</b>
            <button onClick={() => setTweakOpen(false)}>×</button>
          </div>
          <div className="gw-tweaks-body">
            <div className="gw-tw-row">
              <label>Accent hue</label>
              <div className="gw-tw-chips">
                {(['cyan', 'teal', 'amber', 'violet'] as const).map(a => (
                  <button
                    key={a}
                    className={accent === a ? 'on' : ''}
                    style={{ background: ({ cyan: '#4dd6ff', teal: '#00e0c4', amber: '#ffb347', violet: '#a078ff' } as Record<string, string>)[a] }}
                    onClick={() => { setAccent(a); persistTweak('accent', a); }}
                  />
                ))}
              </div>
            </div>
            <div className="gw-tw-row">
              <label>Particle density <em>{density.toFixed(2)}</em></label>
              <input type="range" min="0.2" max="2" step="0.05" value={density}
                onChange={e => { const v = parseFloat(e.target.value); setDensity(v); persistTweak('particleDensity', v); }} />
            </div>
            <div className="gw-tw-row">
              <label>Glow strength <em>{glow.toFixed(2)}</em></label>
              <input type="range" min="0" max="2" step="0.05" value={glow}
                onChange={e => { const v = parseFloat(e.target.value); setGlow(v); persistTweak('glowStrength', v); }} />
            </div>
            <div className="gw-tw-row inline">
              <label>
                <input type="checkbox" checked={showGrid}
                  onChange={e => { setShowGrid(e.target.checked); persistTweak('showGrid', e.target.checked); }} />
                Lat/lon grid
              </label>
            </div>
            <div className="gw-tw-row inline">
              <label>
                <input type="checkbox" checked={showScanlines}
                  onChange={e => { setShowScanlines(e.target.checked); persistTweak('showScanlines', e.target.checked); }} />
                CRT scanlines
              </label>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
