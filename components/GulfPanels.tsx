'use client';

import React, { useMemo } from 'react';
import type { Eddy } from '@/lib/ocean/ssh';
import type { PredictResponse } from '@/lib/databricks/types';
import type { WarmingScenario } from '@/lib/hooks/useWarmingScenarios';

// --- Shared UI primitives ---

interface PanelProps {
  title: string;
  meta?: string;
  children: React.ReactNode;
  className?: string;
}

function Panel({ title, meta, children, className = '' }: PanelProps) {
  return (
    <section className={`gw-panel ${className}`}>
      <header className="gw-panel-h">
        <span className="gw-panel-title">{title}</span>
        {meta && <span className="gw-panel-meta">{meta}</span>}
      </header>
      <div className="gw-panel-body">{children}</div>
    </section>
  );
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  unit?: string;
  hint?: string;
}

function Slider({ label, value, min, max, step, onChange, unit = '', hint }: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="gw-slider">
      <div className="gw-slider-row">
        <span className="gw-slider-label">{label}</span>
        <span className="gw-slider-val">{value.toFixed(step < 1 ? 2 : 0)}{unit}</span>
      </div>
      <div className="gw-slider-track">
        <div className="gw-slider-fill" style={{ width: `${pct}%` }} />
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(parseFloat(e.target.value))} />
      </div>
      {hint && <div className="gw-slider-hint">{hint}</div>}
    </div>
  );
}

interface ToggleProps {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
  swatch?: string;
  desc?: string;
}

function Toggle({ label, value, onChange, swatch, desc }: ToggleProps) {
  return (
    <label className={`gw-toggle ${value ? 'on' : ''}`}>
      <span className="gw-toggle-sw" style={swatch ? { '--sw': swatch } as React.CSSProperties : {}}>
        <span className="gw-toggle-dot" />
      </span>
      <span className="gw-toggle-txt">
        <span>{label}</span>
        {desc && <em>{desc}</em>}
      </span>
      <input type="checkbox" checked={value} onChange={e => onChange(e.target.checked)} />
    </label>
  );
}

// --- LEFT PANEL ---

interface LeftPanelProps {
  t: number;
  setT: (t: number) => void;
  anomaly: number;
  setAnomaly: (a: number) => void;
  loopDepth: number;
  setLoopDepth: (d: number) => void;
  layers: { eddies: boolean; loopCurrent: boolean; riskZones: boolean; predictions: boolean };
  setLayers: (fn: (l: LeftPanelProps['layers']) => LeftPanelProps['layers']) => void;
  speed: number;
  setSpeed: (s: number) => void;
  playing: boolean;
  setPlaying: (fn: (p: boolean) => boolean) => void;
  viewMode: 'past' | 'future';
  setViewMode: (v: 'past' | 'future') => void;
}

export function ViewModeSwitch({ viewMode, setViewMode }: { viewMode: 'past' | 'future'; setViewMode: (v: 'past' | 'future') => void }) {
  return (
    <div className="gw-viewmode-switch" data-view={viewMode}>
      <button className={viewMode === 'past' ? 'on' : ''} onClick={() => setViewMode('past')}>
        Explore the past
      </button>
      <button className={viewMode === 'future' ? 'on' : ''} onClick={() => setViewMode('future')}>
        Project the future
      </button>
      <div className="gw-viewmode-ind" />
    </div>
  );
}

export function LeftPanel({ t, setT, anomaly, setAnomaly, loopDepth, setLoopDepth, layers, setLayers, speed, setSpeed, playing, setPlaying, viewMode, setViewMode }: LeftPanelProps) {
  const yearVal = 1985 + t * 40;
  const year = Math.floor(yearVal);
  const dayOfYear = Math.floor(((yearVal - year) * 365) + 1);
  const date = new Date(year, 0, dayOfYear);
  const dateStr = date.toISOString().slice(0, 10);

  return (
    <aside className="gw-left">
      <ViewModeSwitch viewMode={viewMode} setViewMode={setViewMode} />
      <div className="gw-brand">
        <div className="gw-brand-mark">
          <svg viewBox="0 0 40 40" width="32" height="32">
            <circle cx="20" cy="20" r="17" fill="none" stroke="rgba(140,220,255,0.5)" strokeWidth="1" />
            <circle cx="20" cy="20" r="10" fill="none" stroke="rgba(140,220,255,0.7)" strokeWidth="1" strokeDasharray="2 3" />
            <path d="M 4,20 Q 12,12 20,20 T 36,20" stroke="rgba(255,180,120,0.9)" strokeWidth="1.2" fill="none" />
            <circle cx="20" cy="20" r="2" fill="#4dd6ff" />
          </svg>
        </div>
        <div className="gw-brand-txt">
          <div className="gw-brand-name">GULF WATCH<span className="gw-brand-mk">/</span><em>v2.6</em></div>
          <div className="gw-brand-sub">Gulf Watch — Hurricane Risk Under Climate Change</div>
        </div>
        <div className="gw-brand-status">
          <span className="gw-dot live" /> LIVE
        </div>
      </div>

      <Panel title="DATA SOURCE" meta="NOAA / AVISO · 0.25°">
        <div className="gw-data-readout">
          <div className="gw-data-row"><span>Variable</span><b>Sea Surface Height (cm)</b></div>
          <div className="gw-data-row"><span>Grid</span><b>96 × 56 · daily</b></div>
          <div className="gw-data-row">
            <span>Span</span>
            <b>{viewMode === 'past' ? '1985 → 2025 · historical' : '2025 → 2100 · SSP5-8.5'}</b>
          </div>
          <div className="gw-data-row">
            <span>Frame</span>
            <b className="gw-date-now">
              {viewMode === 'past' ? dateStr : `${Math.floor(2025 + t * 75)} · +${anomaly.toFixed(2)}°C`}
            </b>
          </div>
        </div>
      </Panel>

      <Panel title={viewMode === 'past' ? 'HISTORICAL MODE' : 'CLIMATE PROJECTION'}>
        <div className="gw-mode-hint">
          {viewMode === 'past'
            ? 'Replaying 40 years of Gulf SSH. Eddies & Loop Current traced from SSH gradient.'
            : 'Forward simulation — CNN+LSTM projects LCE separation probability under the warming scenario above.'}
        </div>
      </Panel>

      <Panel title="SIMULATION">
        {viewMode === 'past' ? (
          <Slider label="Time index" value={t} min={0} max={1} step={0.0005}
            onChange={setT} hint={`t = ${(t * 100).toFixed(2)}% of 40-yr window`} />
        ) : (
          <Slider label="Projected year" value={t} min={0} max={1} step={0.0005}
            onChange={setT}
            hint={`${Math.floor(2025 + t * 75)} · ${(t * 100).toFixed(0)}% along 2025→2100 horizon`} />
        )}
        <div className="gw-play-row">
          <button className="gw-play" onClick={() => setPlaying(p => !p)}>
            {playing ? <span>❚❚</span> : <span>▶</span>}
            <em>{playing ? 'PAUSE' : 'PLAY'}</em>
          </button>
          <div className="gw-speed">
            {[1, 2, 5, 10, 20].map(s => (
              <button key={s} className={speed === s ? 'on' : ''} onClick={() => setSpeed(s)}>{s}×</button>
            ))}
          </div>
        </div>
      </Panel>

      <Panel title="OCEAN STATE" meta="Loop Current penetration">
        <Slider
          label="Loop depth"
          value={loopDepth}
          min={0}
          max={1}
          step={0.01}
          onChange={setLoopDepth}
          unit=""
          hint={`${(loopDepth * 100).toFixed(0)}% — ${loopDepth < 0.33 ? 'shallow / attached' : loopDepth < 0.67 ? 'mid penetration' : 'deep / shedding window'}`}
        />
      </Panel>

      <Panel title="FEATURE LAYERS">
        <Toggle label="Eddy detection" desc="local maxima · r ≥ 80 km"
          value={layers.eddies} onChange={v => setLayers(l => ({ ...l, eddies: v }))} swatch="#4dd6ff" />
        <Toggle label="Loop Current" desc="largest high-SSH filament"
          value={layers.loopCurrent} onChange={v => setLayers(l => ({ ...l, loopCurrent: v }))} swatch="#ffb86b" />
        <Toggle label="Risk zones" desc="SSH > 17 cm · rapid intensification"
          value={layers.riskZones} onChange={v => setLayers(l => ({ ...l, riskZones: v }))} swatch="#ff5470" />
        <Toggle label="Predictions vs actual" desc="overlay Δ residuals"
          value={layers.predictions} onChange={v => setLayers(l => ({ ...l, predictions: v }))} swatch="#a078ff" />
      </Panel>

      <div className="gw-footnote">
        Reactive cell graph · {Object.values(layers).filter(Boolean).length + 4} nodes active
      </div>
    </aside>
  );
}

// --- WARMING HERO ---

// Stop positions derived from the inverse accelerated curve
// year = 2025 + 75 * (temp/4)^(1/1.4): temp → year
//   1°C → 2052, 2°C → 2071, 3°C → 2086, 4°C → 2100.
const WARMING_STOPS = [
  { value: 0, label: '2025', sub: '0°C' },
  { value: 1, label: '2052', sub: '+1°C' },
  { value: 2, label: '2071', sub: '+2°C' },
  { value: 3, label: '2086', sub: '+3°C' },
  { value: 4, label: '2100', sub: '+4°C' },
];

interface WarmingHeroProps {
  anomaly: number;
  setAnomaly: (a: number) => void;
  primary?: boolean;
  /** null = past mode (hide link state). true/false = future mode linked state. */
  linked?: boolean | null;
  onRelink?: () => void;
  /** Projected year shown while in future mode. */
  futureYear?: number | null;
}

export function WarmingHero({ anomaly, setAnomaly, primary = false, linked = null, onRelink, futureYear = null }: WarmingHeroProps) {
  const pct = (anomaly / 4) * 100;
  return (
    <div className={`gw-hero-slider${primary ? ' gw-hero-primary' : ''}`}>
      <div className="gw-hero-head">
        <span className="gw-hero-label">
          Gulf warming scenario
          {futureYear !== null && <em className="gw-hero-year"> · {futureYear}</em>}
          {linked === true && <span className="gw-hero-pill gw-hero-pill-auto">AUTO · SSP5-8.5</span>}
          {linked === false && (
            <button className="gw-hero-pill gw-hero-pill-manual" onClick={onRelink}>
              MANUAL · click to re-link
            </button>
          )}
        </span>
        <span className="gw-hero-val">+{anomaly.toFixed(2)} °C</span>
      </div>
      <div className="gw-hero-track">
        <div className="gw-hero-fill" style={{ width: `${pct}%` }} />
        <input
          type="range"
          min={0}
          max={4}
          step={0.05}
          value={anomaly}
          onChange={e => setAnomaly(parseFloat(e.target.value))}
        />
        <div className="gw-hero-stops">
          {WARMING_STOPS.map(s => (
            <button
              key={s.value}
              className={`gw-hero-stop${Math.abs(anomaly - s.value) < 0.04 ? ' on' : ''}`}
              style={{ left: `${(s.value / 4) * 100}%` }}
              onClick={() => setAnomaly(s.value)}
            >
              <i />
              <b>{s.label}</b>
              <em>{s.sub}</em>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// --- RIGHT PANEL ---

interface Region {
  name: string;
  x: number;
  y: number;
  prob: number;
  ssh: string;
  trend: string;
}

interface RightPanelProps {
  t: number;
  anomaly: number;
  grid: Float32Array;
  eddies: Eddy[];
  mode: string;
  onFly?: (r: Region) => void;
  baselinePct: number;
  prediction?: PredictResponse | null;
  warmingScenarios?: WarmingScenario[];
  scenariosLoading?: boolean;
}

export function regionName(x: number, y: number): string {
  if (y > 0.55 && x < 0.45) return 'NW Gulf · Texas shelf';
  if (y > 0.55 && x > 0.55) return 'NE Gulf · Panhandle';
  if (x > 0.6 && y < 0.45) return 'SE Gulf · Florida straits';
  if (x < 0.45 && y < 0.45) return 'Bay of Campeche';
  return 'Central Gulf basin';
}

function deriveModelRegions(prediction: PredictResponse | null | undefined): Region[] {
  if (!prediction) return [];

  const rows = prediction.ri_probability.length;
  const cols = prediction.ri_probability[0]?.length ?? 0;
  if (!rows || !cols) return [];

  const peaks: Array<{ row: number; col: number; prob: number }> = [];
  for (let row = 1; row < rows - 1; row++) {
    for (let col = 1; col < cols - 1; col++) {
      const v = prediction.ri_probability[row][col];
      if (v < 0.12) continue;
      let isPeak = true;
      for (let dr = -1; dr <= 1 && isPeak; dr++) {
        for (let dc = -1; dc <= 1 && isPeak; dc++) {
          if ((dr || dc) && (prediction.ri_probability[row + dr][col + dc] ?? 0) > v) isPeak = false;
        }
      }
      if (isPeak) peaks.push({ row, col, prob: v });
    }
  }

  peaks.sort((a, b) => b.prob - a.prob);
  const kept: typeof peaks = [];
  for (const peak of peaks) {
    if (kept.every(k => Math.hypot(k.row - peak.row, k.col - peak.col) > Math.max(rows, cols) * 0.2)) kept.push(peak);
    if (kept.length >= 3) break;
  }

  return kept.map((peak, index) => {
    const x = peak.col / Math.max(1, cols - 1);
    const y = 1 - peak.row / Math.max(1, rows - 1);
    return {
      name: index === 0 ? prediction.highest_risk_zone : regionName(x, y),
      x,
      y,
      prob: peak.prob,
      ssh: (peak.prob * 17).toFixed(1),
      trend: peak.prob >= prediction.lce_separation_prob_30d ? '▲' : '▼',
    };
  });
}

export function RightPanel({ t, anomaly, grid, eddies, mode, onFly, baselinePct, prediction, warmingScenarios, scenariosLoading }: RightPanelProps) {
  const fallbackRegions = useMemo<Region[]>(() => {
    return eddies
      .map(e => ({
        name: regionName(e.x, e.y),
        x: e.x, y: e.y,
        prob: Math.min(0.99, (e.strength - 0.35) * 1.15 + anomaly * 0.08),
        ssh: (e.strength * 17 + 8).toFixed(1),
        trend: Math.sin(t * 30 + e.x * 9) > 0 ? '▲' : '▼',
      }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 3);
  }, [eddies, t, anomaly]);
  const modelRegions = useMemo<Region[]>(() => deriveModelRegions(prediction), [prediction]);
  const regions = modelRegions.length > 0 ? modelRegions : fallbackRegions;

  const p7 = prediction?.lce_separation_prob_7d ?? 0;
  const p30 = prediction?.lce_separation_prob_30d ?? 0;
  const days = prediction?.ri_days_per_year ?? 0;

  // Real scenario data from 4 parallel predict calls at sst_delta = 0,1,2,3.
  // This is what the MODEL actually says — no client-side extrapolation.
  const scenarios = useMemo(() => {
    if (!warmingScenarios) return [];
    return warmingScenarios.map(s => ({
      c: s.sst_delta,
      p7: s.data?.lce_separation_prob_7d ?? 0,
      p30: s.data?.lce_separation_prob_30d ?? 0,
      days: s.data?.ri_days_per_year ?? 0,
      loaded: !!s.data,
    }));
  }, [warmingScenarios]);
  const peakDays = Math.max(...scenarios.map(s => s.days), 1);
  const scenario0 = scenarios.find(s => s.c === 0);
  const scenarioMax = scenarios.find(s => s.c === 4);
  // Detect whether the model's LCE heads actually respond to warming. If 7d
  // spread across +0..+4°C is below 0.5 pp, flag as flat so the panel can
  // tell judges the truth instead of implying a linear sensitivity.
  const p7Spread = scenario0 && scenarioMax ? Math.abs(scenarioMax.p7 - scenario0.p7) : 0;
  const p30Spread = scenario0 && scenarioMax ? Math.abs(scenarioMax.p30 - scenario0.p30) : 0;
  const daysSpread = scenario0 && scenarioMax ? scenarioMax.days - scenario0.days : 0;
  const separationFlat = p7Spread < 0.005 && p30Spread < 0.005;

  return (
    <aside className="gw-right">
      <Panel
        title="MODEL PREDICTIONS"
        meta={prediction ? (prediction.source === 'stub' ? 'DEMO DATA' : 'LIVE · gulf-watch-v1') : 'connecting…'}
      >
        <div className="gw-pred-head">
          <div className="gw-pred-current">
            <span>Current scenario</span>
            <b>+{anomaly.toFixed(1)}°C</b>
            <em>{mode === 'predicted' ? 'Climate projection' : 'Historical replay'}</em>
          </div>
          <div className="gw-pred-ri">
            <span>RI days / yr</span>
            <b>{prediction ? days.toFixed(1) : '—'}</b>
          </div>
        </div>

        <div className="gw-pred-bar">
          <div className="gw-pred-bar-row">
            <span className="gw-pred-bar-lbl">P(LCE separation · 7d)</span>
            <span className="gw-pred-bar-val">{prediction ? `${(p7 * 100).toFixed(0)}%` : '—'}</span>
          </div>
          <div className="gw-pred-bar-track">
            <div className="gw-pred-bar-fill gw-pred-7d" style={{ width: `${p7 * 100}%` }} />
          </div>
        </div>
        <div className="gw-pred-bar">
          <div className="gw-pred-bar-row">
            <span className="gw-pred-bar-lbl">P(LCE separation · 30d)</span>
            <span className="gw-pred-bar-val">{prediction ? `${(p30 * 100).toFixed(0)}%` : '—'}</span>
          </div>
          <div className="gw-pred-bar-track">
            <div className="gw-pred-bar-fill gw-pred-30d" style={{ width: `${p30 * 100}%` }} />
          </div>
        </div>

        <div className="gw-pred-hint">
          Two sigmoid heads of the CNN+LSTM. 7-day head captures imminent eddy detachment; 30-day head captures the fuller separation cycle.
        </div>
      </Panel>

      <Panel title="WARMING SENSITIVITY" meta={scenariosLoading ? 'running 4 model calls…' : 'measured · 4 predict calls'}>
        <div className="gw-sens-grid">
          <div className="gw-sens-h">
            <span>°C</span>
            <span>7d</span>
            <span>30d</span>
            <span>RI d/yr</span>
          </div>
          {scenarios.map(s => {
            const isActive = Math.abs(s.c - anomaly) < 0.5;
            return (
              <div key={s.c} className={`gw-sens-row${isActive ? ' on' : ''}`}>
                <span className="gw-sens-c">+{s.c}°C</span>
                <span className="gw-sens-cell">
                  <i className="gw-sens-bar" style={{ width: `${s.p7 * 100}%`, background: '#4dd6ff' }} />
                  <em>{(s.p7 * 100).toFixed(0)}%</em>
                </span>
                <span className="gw-sens-cell">
                  <i className="gw-sens-bar" style={{ width: `${s.p30 * 100}%`, background: '#a078ff' }} />
                  <em>{(s.p30 * 100).toFixed(0)}%</em>
                </span>
                <span className="gw-sens-cell">
                  <i className="gw-sens-bar" style={{ width: `${(s.days / peakDays) * 100}%`, background: '#ff8a6b' }} />
                  <em>{s.days.toFixed(1)}</em>
                </span>
              </div>
            );
          })}
        </div>
        <div className="gw-pred-hint">
          {scenariosLoading && scenarios.every(s => !s.loaded)
            ? <>Querying model at 5 warming levels…</>
            : separationFlat
              ? <>Separation heads return <b>flat</b> across +0 to +4°C (Δ &lt; 0.5pp) — this model's LCE probability reads SSH geometry only, not SST. RI days/yr responds: +<b>{daysSpread.toFixed(1)}</b> days from +0 → +4°C.</>
              : <>Measured across +0 → +4°C (SSP5-8.5 end-of-century): 7d separation shifts <b>{(p7Spread * 100).toFixed(1)} pp</b>, 30d shifts <b>{(p30Spread * 100).toFixed(1)} pp</b>, RI days/yr adds <b>{daysSpread.toFixed(1)}</b>.</>}
        </div>
      </Panel>

      <Panel title="TOP RISK ZONES" meta={prediction ? `${regions.length} flagged · model heatmap` : `${regions.length} flagged · SSH surrogate`}>
        <div className="gw-regions">
          {regions.map((r, i) => {
            // Use the real sensitivity measured from the scenarios endpoint
            // instead of a hard-coded slope. Falls back to no-delta if
            // scenarios haven't loaded yet.
            const measuredSlope = scenario0 && scenarioMax ? (scenarioMax.p7 - scenario0.p7) / 4 : 0;
            const baseProb = Math.max(0, r.prob - measuredSlope * anomaly);
            const delta = (r.prob - baseProb) * 100;
            return (
              <div className="gw-region" key={i} onClick={() => onFly?.(r)}>
                <div className="gw-region-rank">0{i + 1}</div>
                <div className="gw-region-body">
                  <div className="gw-region-name">{r.name}</div>
                  <div className="gw-region-sub">lat {(18 + r.y * 12).toFixed(1)}°N · lon {(98 - r.x * 18).toFixed(1)}°W</div>
                  <div className="gw-region-bar">
                    <div className="gw-region-fill" style={{ width: `${r.prob * 100}%` }} />
                    <div className="gw-region-bar-base" style={{ width: `${baseProb * 100}%` }} />
                  </div>
                  <div className="gw-region-delta">
                    baseline <b>{(baseProb * 100) | 0}%</b> → at +{anomaly.toFixed(1)}°C <b>{(r.prob * 100) | 0}%</b>
                    <em className={delta > 0 ? 'up' : 'dn'}>+{delta.toFixed(0)}pp</em>
                  </div>
                </div>
              </div>
            );
          })}
          {regions.length === 0 && <div className="gw-empty">No regions above the model&apos;s risk threshold.</div>}
        </div>
      </Panel>
    </aside>
  );
}

// --- BOTTOM ANALYTICS ---

interface AnalyticsProps {
  tSeries: number[];
  eddySeries: number[];
  tIdx: number;
}

export function Analytics({ tSeries, eddySeries, tIdx }: AnalyticsProps) {
  function Sparkline({ data, color, label, value, unit, range }: {
    data: number[]; color: string; label: string; value: string; unit: string; range?: [number, number];
  }) {
    const w = 240, h = 48;
    const [lo, hi] = range ?? [Math.min(...data), Math.max(...data)];
    const safe = (v: number) => isFinite(v) ? v : 0;
    const path = data.map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = safe(h - ((v - lo) / (hi - lo || 1)) * (h - 4) - 2);
      return `${i ? 'L' : 'M'} ${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    const cursor = Math.min(data.length - 1, Math.max(0, Math.floor(tIdx * (data.length - 1)) || 0));
    const cx = safe((cursor / (data.length - 1)) * w);
    const cy = safe(h - ((data[cursor] - lo) / (hi - lo || 1)) * (h - 4) - 2);
    return (
      <div className="gw-spark">
        <div className="gw-spark-h">
          <span>{label}</span>
          <b style={{ color }}>{value}{unit}</b>
        </div>
        <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} preserveAspectRatio="none">
          <defs>
            <linearGradient id={`g${label}`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.4" />
              <stop offset="100%" stopColor={color} stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={`${path} L ${w},${h} L 0,${h} Z`} fill={`url(#g${label})`} />
          <path d={path} stroke={color} strokeWidth="1.2" fill="none" />
          <line x1={cx} x2={cx} y1="0" y2={h} stroke="rgba(255,255,255,0.35)" strokeWidth="0.8" strokeDasharray="2 2" />
          <circle cx={cx} cy={cy} r="2.5" fill={color} />
        </svg>
      </div>
    );
  }

  function LegendSpark() {
    return (
      <div className="gw-spark gw-spark-legend-cell">
        <div className="gw-spark-h">
          <span>SSH ANOMALY · cm</span>
          <b style={{ color: '#ff8aa2' }}>RI &gt; +17 cm</b>
        </div>
        <div className="gw-legend-bar">
          <span className="gw-legend-grad" />
          <span className="gw-legend-ticks">
            <i>−20</i><i>−10</i><i>0</i><i>+10</i><i>+17</i><i>+25</i>
          </span>
        </div>
        <div className="gw-legend-note">
          <span className="gw-pip risk" /> colormap of the model&apos;s RI field · red zones = rapid intensification potential
        </div>
      </div>
    );
  }

  return (
    <div className="gw-analytics">
      <Sparkline data={tSeries} color="#4dd6ff" label="SSH anomaly"
        value={tSeries[Math.floor(tIdx * (tSeries.length - 1))].toFixed(2)} unit=" cm"
        range={[-10, 25]} />
      <Sparkline data={eddySeries} color="#ffb347" label="Eddy count"
        value={eddySeries[Math.floor(tIdx * (eddySeries.length - 1))].toFixed(0)} unit=""
        range={[0, 12]} />
      <LegendSpark />
    </div>
  );
}
