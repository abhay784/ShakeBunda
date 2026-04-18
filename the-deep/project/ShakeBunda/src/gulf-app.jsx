/* global React, ReactDOM, GulfMap, LeftPanel, RightPanel, Analytics, Timeline, buildGrid, detectEddies, EVENTS */

const { useState, useEffect, useRef, useMemo } = React;

function App() {
  const TW = window.__TWEAKS__ || {};
  const [t, setT] = useState(() => {
    const saved = parseFloat(localStorage.getItem('gw.t') || '');
    return isNaN(saved) ? (2021 - 1985) / 40 : saved;
  });
  const [anomaly, setAnomaly] = useState(TW.tempAnomaly ?? 0.9);
  const [layers, setLayers] = useState({
    eddies: true, loopCurrent: true, riskZones: true, predictions: false,
  });
  const [mode, setMode] = useState(TW.mode || 'historical');
  const [speed, setSpeed] = useState(5);
  const [playing, setPlaying] = useState(false);
  const [eddyCount, setEddyCount] = useState(0);
  const [tweakOpen, setTweakOpen] = useState(false);
  const [accent, setAccent] = useState(TW.accent || 'cyan');
  const [density, setDensity] = useState(TW.particleDensity ?? 1.0);
  const [glow, setGlow] = useState(TW.glowStrength ?? 1.0);
  const [showGrid, setShowGrid] = useState(TW.showGrid !== false);
  const [showScanlines, setShowScanlines] = useState(TW.showScanlines !== false);

  // persist time
  useEffect(() => { localStorage.setItem('gw.t', t.toString()); }, [t]);

  // playback
  useEffect(() => {
    if (!playing) return;
    let raf, last = performance.now();
    const loop = (now) => {
      const dt = (now - last) / 1000; last = now;
      // 1× = 1 year per 40s of real time. Tunable.
      setT(prev => {
        const nt = prev + (dt / 40) * (speed / 1);
        return nt > 1 ? 0 : nt;
      });
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed]);

  // derived grid / eddies (cheap; GulfMap also builds its own for rendering)
  const grid = useMemo(() => buildGrid(t, anomaly), [t, anomaly]);
  const eddies = useMemo(() => detectEddies(grid), [grid]);

  // time series for analytics — sample grid across recent window
  const [series, setSeries] = useState(() => genSeries());
  function genSeries() {
    const ts = [], es = [], ps = [];
    const N = 120;
    for (let i = 0; i < N; i++) {
      const tt = i / (N - 1);
      const g = buildGrid(tt, 0.8);
      // mean SSH (cm)
      let sum = 0, cnt = 0, peaks = 0;
      for (let k = 0; k < g.length; k++) {
        if (!isNaN(g[k])) { sum += g[k]; cnt++; }
      }
      ts.push((sum/cnt) * 17);
      es.push(detectEddies(g).length);
      ps.push((sum/cnt) * 17 + (Math.sin(tt*30)*0.6 + (Math.random()-0.5)*0.4));
    }
    return { ts, es, ps };
  }

  // Tweaks listener
  useEffect(() => {
    const onMsg = (e) => {
      if (!e.data) return;
      if (e.data.type === '__activate_edit_mode') setTweakOpen(true);
      if (e.data.type === '__deactivate_edit_mode') setTweakOpen(false);
    };
    window.addEventListener('message', onMsg);
    window.parent.postMessage({ type: '__edit_mode_available' }, '*');
    return () => window.removeEventListener('message', onMsg);
  }, []);

  function persistTweak(key, val) {
    window.parent.postMessage({ type: '__edit_mode_set_keys', edits: { [key]: val } }, '*');
  }

  // flyTo: pulse a circle on map area
  const [pin, setPin] = useState(null);
  function flyTo(r) {
    setPin(r);
    setTimeout(() => setPin(null), 2800);
  }

  const accentHex = {
    cyan: '#4dd6ff', teal: '#00e0c4', amber: '#ffb347', violet: '#a078ff'
  }[accent] || '#4dd6ff';

  return (
    <div className={`gw-app ${showScanlines?'scan':''}`} style={{'--accent': accentHex}}>
      <LeftPanel
        t={t} setT={setT}
        anomaly={anomaly} setAnomaly={setAnomaly}
        layers={layers} setLayers={setLayers}
        mode={mode} setMode={setMode}
        speed={speed} setSpeed={setSpeed}
        playing={playing} setPlaying={setPlaying}
      />

      <main className="gw-main">
        {/* Top status bar */}
        <div className="gw-topbar">
          <div className="gw-top-left">
            <span className="gw-breadcrumb active">GULF · 18–31°N · 98–80°W</span>
            <span className="gw-top-chip"><i className="gw-dot live"/>{mode==='predicted'?'ORM-v3':'HINDCAST'}</span>
          </div>
          <div className="gw-top-right">
            <span className="gw-top-stat">{eddyCount} eddies · {(Math.max(...eddies.map(e=>e.strength*17+8), 0)).toFixed(1)}cm peak</span>
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
            mode={mode}
          />

          {/* Overlay chrome */}
          <div className="gw-stage-chrome">
            <div className="gw-legend">
              <div className="gw-legend-h">SSH ANOMALY · cm</div>
              <div className="gw-legend-bar">
                <span className="gw-legend-grad"/>
                <span className="gw-legend-ticks">
                  <i>−20</i><i>−10</i><i>0</i><i>+10</i><i>+17</i><i>+25</i>
                </span>
              </div>
              <div className="gw-legend-note">
                <span className="gw-pip risk"/> SSH &gt; +17 cm · <b>rapid intensification potential</b>
              </div>
            </div>

            <div className="gw-crosshair">
              <div className="gw-crosshair-h"/>
              <div className="gw-crosshair-v"/>
              <div className="gw-crosshair-readout">
                <b>26.4°N · 88.2°W</b>
                <span>SSH <i>{(Math.sin(t*30)*8+12).toFixed(1)} cm</i></span>
                <span>SST <i>{(28.4 + anomaly).toFixed(1)} °C</i></span>
                <span>UOHC <i>{(85 + anomaly*18).toFixed(0)} kJ/cm²</i></span>
              </div>
            </div>

            {pin && (
              <div className="gw-pin" style={{left: `${pin.x*100}%`, top: `${(1-pin.y)*100}%`}}>
                <div className="gw-pin-ring"/>
                <div className="gw-pin-ring d2"/>
                <div className="gw-pin-lbl">{pin.name} · {(pin.prob*100|0)}%</div>
              </div>
            )}

            <div className="gw-stage-meta">
              <span>FRAME {((t*14610)|0).toString().padStart(5,'0')} / 14610</span>
              <span>· AVISO L4 DUACS-2021 · 0.25°</span>
              <span>· {playing ? `▶ ${speed}×` : '❚❚ paused'}</span>
            </div>
          </div>
        </div>

        <div className="gw-bottom">
          <Analytics tSeries={series.ts} eddySeries={series.es} predSeries={series.ps}
                     tIdx={t}/>
          <Timeline t={t} setT={setT} playing={playing}/>
        </div>
      </main>

      <RightPanel
        t={t} anomaly={anomaly} grid={grid} eddies={eddies} mode={mode}
        onFly={flyTo}
      />

      {tweakOpen && (
        <div className="gw-tweaks">
          <div className="gw-tweaks-h">
            <b>TWEAKS</b>
            <button onClick={()=>setTweakOpen(false)}>×</button>
          </div>
          <div className="gw-tweaks-body">
            <div className="gw-tw-row">
              <label>Accent hue</label>
              <div className="gw-tw-chips">
                {['cyan','teal','amber','violet'].map(a => (
                  <button key={a} className={accent===a?'on':''}
                          style={{background: {cyan:'#4dd6ff',teal:'#00e0c4',amber:'#ffb347',violet:'#a078ff'}[a]}}
                          onClick={()=>{setAccent(a); persistTweak('accent', a);}}/>
                ))}
              </div>
            </div>
            <div className="gw-tw-row">
              <label>Particle density <em>{density.toFixed(2)}</em></label>
              <input type="range" min="0.2" max="2" step="0.05" value={density}
                     onChange={e=>{const v=parseFloat(e.target.value); setDensity(v); persistTweak('particleDensity', v);}}/>
            </div>
            <div className="gw-tw-row">
              <label>Glow strength <em>{glow.toFixed(2)}</em></label>
              <input type="range" min="0" max="2" step="0.05" value={glow}
                     onChange={e=>{const v=parseFloat(e.target.value); setGlow(v); persistTweak('glowStrength', v);}}/>
            </div>
            <div className="gw-tw-row inline">
              <label><input type="checkbox" checked={showGrid}
                     onChange={e=>{setShowGrid(e.target.checked); persistTweak('showGrid', e.target.checked);}}/> Lat/lon grid</label>
            </div>
            <div className="gw-tw-row inline">
              <label><input type="checkbox" checked={showScanlines}
                     onChange={e=>{setShowScanlines(e.target.checked); persistTweak('showScanlines', e.target.checked);}}/> CRT scanlines</label>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// keyboard
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const ev = new CustomEvent('gw-key', { detail: e.key });
  window.dispatchEvent(ev);
});

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
