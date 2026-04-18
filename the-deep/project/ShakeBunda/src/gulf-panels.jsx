/* global React */
// Left / Right / Analytics panels

const { useEffect, useState, useMemo, useRef } = React;

// --- shared UI primitives ---
function Panel({ title, meta, children, className = "" }) {
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

function Slider({ label, value, min, max, step, onChange, unit = "", hint }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="gw-slider">
      <div className="gw-slider-row">
        <span className="gw-slider-label">{label}</span>
        <span className="gw-slider-val">{typeof value === 'number' ? value.toFixed(step < 1 ? 2 : 0) : value}{unit}</span>
      </div>
      <div className="gw-slider-track">
        <div className="gw-slider-fill" style={{width: `${pct}%`}}/>
        <input type="range" min={min} max={max} step={step} value={value}
               onChange={e => onChange(parseFloat(e.target.value))}/>
      </div>
      {hint && <div className="gw-slider-hint">{hint}</div>}
    </div>
  );
}

function Toggle({ label, value, onChange, swatch, desc }) {
  return (
    <label className={`gw-toggle ${value ? 'on' : ''}`}>
      <span className="gw-toggle-sw" style={swatch ? {'--sw': swatch} : {}}>
        <span className="gw-toggle-dot"/>
      </span>
      <span className="gw-toggle-txt">
        <span>{label}</span>
        {desc && <em>{desc}</em>}
      </span>
      <input type="checkbox" checked={value} onChange={e => onChange(e.target.checked)}/>
    </label>
  );
}

// --- LEFT PANEL ---
function LeftPanel({
  t, setT,
  anomaly, setAnomaly,
  layers, setLayers,
  mode, setMode,
  speed, setSpeed,
  playing, setPlaying,
}) {
  const yearVal = 1985 + t * 40;
  const year = Math.floor(yearVal);
  const dayOfYear = Math.floor(((yearVal - year) * 365) + 1);
  const date = new Date(year, 0, dayOfYear);
  const dateStr = date.toISOString().slice(0,10);

  return (
    <aside className="gw-left">
      <div className="gw-brand">
        <div className="gw-brand-mark">
          <svg viewBox="0 0 40 40" width="32" height="32">
            <circle cx="20" cy="20" r="17" fill="none" stroke="rgba(140,220,255,0.5)" strokeWidth="1"/>
            <circle cx="20" cy="20" r="10" fill="none" stroke="rgba(140,220,255,0.7)" strokeWidth="1" strokeDasharray="2 3"/>
            <path d="M 4,20 Q 12,12 20,20 T 36,20" stroke="rgba(255,180,120,0.9)" strokeWidth="1.2" fill="none"/>
            <circle cx="20" cy="20" r="2" fill="#4dd6ff"/>
          </svg>
        </div>
        <div className="gw-brand-txt">
          <div className="gw-brand-name">GULF WATCH<span className="gw-brand-mk">/</span><em>v2.6</em></div>
          <div className="gw-brand-sub">Ocean Risk Observation Console</div>
        </div>
        <div className="gw-brand-status">
          <span className="gw-dot live"/> LIVE
        </div>
      </div>

      <Panel title="DATA SOURCE" meta="NOAA / AVISO · 0.25°">
        <div className="gw-data-readout">
          <div className="gw-data-row"><span>Variable</span><b>Sea Surface Height (cm)</b></div>
          <div className="gw-data-row"><span>Grid</span><b>96 × 56 · daily</b></div>
          <div className="gw-data-row"><span>Span</span><b>1985 · 01 · 01 → 2025 · 12 · 31</b></div>
          <div className="gw-data-row"><span>Frame</span><b className="gw-date-now">{dateStr}</b></div>
        </div>
      </Panel>

      <Panel title="MODE">
        <div className="gw-mode-switch" data-mode={mode}>
          <button className={mode==='historical'?'on':''} onClick={()=>setMode('historical')}>
            <span className="gw-mode-glyph">◉</span> Historical
          </button>
          <button className={mode==='predicted'?'on':''} onClick={()=>setMode('predicted')}>
            <span className="gw-mode-glyph">◈</span> Predicted (AI)
          </button>
          <div className="gw-mode-ind"/>
        </div>
        <div className="gw-mode-hint">
          {mode === 'historical'
            ? 'Replaying observed AVISO altimetry. Eddies & Loop Current traced from SSH gradient.'
            : 'Forward simulation — SSH field evolved via convolutional ocean-surrogate model (τ+14d).'}
        </div>
      </Panel>

      <Panel title="SIMULATION">
        <Slider label="Time index" value={t} min={0} max={1} step={0.0005}
                onChange={setT} unit="" hint={`t = ${(t*100).toFixed(2)}% of 40-yr window`}/>
        <Slider label="Temperature anomaly" value={anomaly} min={0} max={3} step={0.05}
                onChange={setAnomaly} unit=" °C"
                hint={anomaly > 1.8 ? '⚠ amplifies SSH by ' + (anomaly*0.55*100|0) + '%' : `amplifies SSH by ${(anomaly*0.55*100|0)}%`}/>
        <div className="gw-play-row">
          <button className="gw-play" onClick={()=>setPlaying(p=>!p)}>
            {playing ? <span>❚❚</span> : <span>▶</span>}
            <em>{playing ? 'PAUSE' : 'PLAY'}</em>
          </button>
          <div className="gw-speed">
            {[1,2,5,10,20].map(s => (
              <button key={s} className={speed===s?'on':''} onClick={()=>setSpeed(s)}>{s}×</button>
            ))}
          </div>
        </div>
      </Panel>

      <Panel title="FEATURE LAYERS">
        <Toggle label="Eddy detection" desc="local maxima · r ≥ 80 km"
                value={layers.eddies} onChange={v=>setLayers(l=>({...l, eddies:v}))} swatch="#4dd6ff"/>
        <Toggle label="Loop Current" desc="largest high-SSH filament"
                value={layers.loopCurrent} onChange={v=>setLayers(l=>({...l, loopCurrent:v}))} swatch="#ffb86b"/>
        <Toggle label="Risk zones" desc="SSH > 17 cm · rapid intensification"
                value={layers.riskZones} onChange={v=>setLayers(l=>({...l, riskZones:v}))} swatch="#ff5470"/>
        <Toggle label="Predictions vs actual" desc="overlay Δ residuals"
                value={layers.predictions} onChange={v=>setLayers(l=>({...l, predictions:v}))} swatch="#a078ff"/>
      </Panel>

      <div className="gw-footnote">
        Reactive cell graph · {Object.values(layers).filter(Boolean).length + 4} nodes active
      </div>
    </aside>
  );
}

// --- RIGHT PANEL: AI Ocean Risk Engine ---
function RightPanel({ t, anomaly, grid, eddies, mode, onFly }) {
  const [chat, setChat] = useState([
    {role:'sys', text:'Ocean Risk Engine initialized. SSH surrogate ORM-v3 loaded.'},
  ]);
  const [input, setInput] = useState('Where would a storm intensify today?');
  const chatEnd = useRef(null);

  useEffect(() => { chatEnd.current?.scrollTo(0, 99999); }, [chat]);

  // derive top 3 risk regions from eddies + SSH
  const regions = useMemo(() => {
    const cand = eddies.map(e => ({
      name: regionName(e.x, e.y),
      x: e.x, y: e.y,
      prob: Math.min(0.99, (e.strength - 0.35) * 1.15 + anomaly*0.08),
      ssh: (e.strength*17 + 8).toFixed(1),
      trend: Math.sin(t*30 + e.x*9) > 0 ? '▲' : '▼',
    })).sort((a,b)=>b.prob-a.prob).slice(0,3);
    return cand;
  }, [eddies, t, anomaly]);

  function regionName(x, y) {
    if (y > 0.55 && x < 0.45) return 'NW Gulf · Texas shelf';
    if (y > 0.55 && x > 0.55) return 'NE Gulf · Panhandle';
    if (x > 0.6 && y < 0.45)  return 'SE Gulf · Florida straits';
    if (x < 0.45 && y < 0.45) return 'Bay of Campeche';
    return 'Central Gulf basin';
  }

  function ask() {
    if (!input.trim()) return;
    const q = input.trim();
    setChat(c => [...c, {role:'u', text:q}]);
    setInput('');
    // simulated reasoning
    setTimeout(() => {
      const top = regions[0];
      if (!top) {
        setChat(c => [...c, {role:'a', text:'No high-SSH anomaly exceeds the +14 cm threshold in the current frame.'}]);
        return;
      }
      const lines = [
        `Highest intensification potential: ${top.name} (${(top.prob*100|0)}% likelihood).`,
        `SSH anomaly there is ${top.ssh} cm — the upper-ocean heat content can sustain a Cat-${Math.min(5, 3+Math.floor(top.prob*2))} storm.`,
        `Flagged ${regions.length} zones above threshold. Pinning ${top.name} on the map.`,
      ];
      setChat(c => [...c, {role:'a', text: lines.join(' '), pin: top}]);
      onFly && onFly(top);
    }, 420);
  }

  return (
    <aside className="gw-right">
      <Panel title="OCEAN RISK ENGINE" meta={mode==='predicted'?'FORECAST · τ+14d':'HINDCAST'}>
        <div className="gw-risk-hero">
          <div className="gw-risk-gauge">
            <svg viewBox="0 0 120 60" width="100%" height="80">
              <defs>
                <linearGradient id="gauge" x1="0" x2="1">
                  <stop offset="0%" stopColor="#4dd6ff"/>
                  <stop offset="55%" stopColor="#ffb347"/>
                  <stop offset="100%" stopColor="#ff3864"/>
                </linearGradient>
              </defs>
              <path d="M 10,55 A 50,50 0 0 1 110,55" stroke="url(#gauge)" strokeWidth="4" fill="none" strokeLinecap="round"/>
              <path d="M 10,55 A 50,50 0 0 1 110,55" stroke="rgba(255,255,255,0.06)" strokeWidth="12" fill="none"/>
              {(() => {
                const p = Math.min(1, (regions[0]?.prob || 0));
                const a = Math.PI * (1 - p);
                const nx = 60 + Math.cos(a)*48, ny = 55 - Math.sin(a)*48;
                return <g>
                  <line x1="60" y1="55" x2={nx} y2={ny} stroke="#fff" strokeWidth="1.5"/>
                  <circle cx="60" cy="55" r="4" fill="#fff"/>
                </g>;
              })()}
            </svg>
            <div className="gw-gauge-lbl">
              <b>{((regions[0]?.prob || 0)*100).toFixed(0)}%</b>
              <span>Basin-wide RI probability · next 72h</span>
            </div>
          </div>
        </div>
      </Panel>

      <Panel title="TOP 3 RISK REGIONS" meta={`${regions.length} flagged`}>
        <div className="gw-regions">
          {regions.map((r, i) => (
            <div className="gw-region" key={i} onClick={()=>onFly && onFly(r)}>
              <div className="gw-region-rank">0{i+1}</div>
              <div className="gw-region-body">
                <div className="gw-region-name">{r.name}</div>
                <div className="gw-region-sub">Δh <b>{r.ssh} cm</b> · lat {(18+r.y*12).toFixed(1)}°N · lon {(98-r.x*18).toFixed(1)}°W</div>
                <div className="gw-region-bar">
                  <div className="gw-region-fill" style={{width: `${r.prob*100}%`}}/>
                </div>
              </div>
              <div className="gw-region-pct">
                <b>{(r.prob*100|0)}</b><em>%</em>
                <span className={r.trend==='▲'?'up':'dn'}>{r.trend}</span>
              </div>
            </div>
          ))}
          {regions.length === 0 && <div className="gw-empty">No regions above +14 cm threshold.</div>}
        </div>
      </Panel>

      <Panel title="AI INTERACTION" meta="ORM-v3 · local">
        <div className="gw-chat" ref={chatEnd}>
          {chat.map((m, i) => (
            <div key={i} className={`gw-msg gw-msg-${m.role}`}>
              {m.role === 'sys' && <span className="gw-msg-tag">SYS</span>}
              {m.role === 'u' && <span className="gw-msg-tag">YOU</span>}
              {m.role === 'a' && <span className="gw-msg-tag">ORM</span>}
              <span>{m.text}</span>
              {m.pin && <span className="gw-msg-pin">📍 pinned {m.pin.name}</span>}
            </div>
          ))}
        </div>
        <form className="gw-chat-input" onSubmit={e=>{e.preventDefault(); ask();}}>
          <span className="gw-chat-prompt">&gt;</span>
          <input value={input} onChange={e=>setInput(e.target.value)} placeholder="ask the ocean…"/>
          <button type="submit">↵</button>
        </form>
        <div className="gw-chat-suggest">
          {['Why is SSH high near Tampa?', 'Compare 2005 to 2017', 'Predict eddy shedding'].map(s =>
            <button key={s} onClick={()=>setInput(s)}>{s}</button>
          )}
        </div>
      </Panel>
    </aside>
  );
}

// --- BOTTOM ANALYTICS ---
function Analytics({ tSeries, eddySeries, predSeries, tIdx }) {
  function Sparkline({ data, color, label, value, unit, range }) {
    const w = 240, h = 48;
    const [lo, hi] = range || [Math.min(...data), Math.max(...data)];
    const safe = (v) => isFinite(v) ? v : 0;
    const path = data.map((v,i) => {
      const x = (i/(data.length-1))*w;
      const y = safe(h - ((v-lo)/(hi-lo||1))*(h-4) - 2);
      return `${i?'L':'M'} ${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    const cursor = Math.min(data.length-1, Math.max(0, Math.floor(tIdx * (data.length-1)) || 0));
    const cx = safe((cursor/(data.length-1))*w);
    const cy = safe(h - ((data[cursor]-lo)/(hi-lo||1))*(h-4) - 2);
    return (
      <div className="gw-spark">
        <div className="gw-spark-h">
          <span>{label}</span>
          <b style={{color}}>{value}{unit}</b>
        </div>
        <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} preserveAspectRatio="none">
          <defs>
            <linearGradient id={`g${label}`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.4"/>
              <stop offset="100%" stopColor={color} stopOpacity="0"/>
            </linearGradient>
          </defs>
          <path d={`${path} L ${w},${h} L 0,${h} Z`} fill={`url(#g${label})`}/>
          <path d={path} stroke={color} strokeWidth="1.2" fill="none"/>
          <line x1={cx} x2={cx} y1="0" y2={h} stroke="rgba(255,255,255,0.35)" strokeWidth="0.8" strokeDasharray="2 2"/>
          <circle cx={cx} cy={cy} r="2.5" fill={color}/>
        </svg>
      </div>
    );
  }

  // pred vs actual: 2 lines
  function DualSpark({ a, b, label }) {
    const w = 240, h = 48;
    const lo = Math.min(...a, ...b), hi = Math.max(...a, ...b);
    const toPath = (arr) => arr.map((v,i)=>{
      const x = (i/(arr.length-1))*w;
      const y = h - ((v-lo)/(hi-lo||1))*(h-4) - 2;
      return `${i?'L':'M'} ${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    const err = a.map((v,i)=> Math.abs(v-b[i])).reduce((p,c)=>p+c,0) / a.length;
    return (
      <div className="gw-spark">
        <div className="gw-spark-h">
          <span>{label}</span>
          <b style={{color: '#a078ff'}}>MAE {err.toFixed(3)}</b>
        </div>
        <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} preserveAspectRatio="none">
          <path d={toPath(a)} stroke="#4dd6ff" strokeWidth="1.2" fill="none"/>
          <path d={toPath(b)} stroke="#a078ff" strokeWidth="1.2" fill="none" strokeDasharray="3 2"/>
        </svg>
        <div className="gw-spark-legend">
          <span><i style={{background:'#4dd6ff'}}/>observed</span>
          <span><i style={{background:'#a078ff'}}/>predicted</span>
        </div>
      </div>
    );
  }

  return (
    <div className="gw-analytics">
      <Sparkline data={tSeries} color="#4dd6ff" label="SSH anomaly"
                 value={tSeries[Math.floor(tIdx*(tSeries.length-1))].toFixed(2)} unit=" cm"
                 range={[-10, 25]}/>
      <Sparkline data={eddySeries} color="#ffb347" label="Eddy count"
                 value={eddySeries[Math.floor(tIdx*(eddySeries.length-1))].toFixed(0)} unit=""
                 range={[0, 12]}/>
      <DualSpark a={tSeries} b={predSeries} label="Predicted vs actual"/>
    </div>
  );
}

window.LeftPanel = LeftPanel;
window.RightPanel = RightPanel;
window.Analytics = Analytics;
window.Panel = Panel;
window.Slider = Slider;
window.Toggle = Toggle;
