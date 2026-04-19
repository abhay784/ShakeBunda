/* global React */
// Gulf of Mexico SSH heatmap + particle flow simulation
// Data is SCALAR SSH. We derive a pseudo-flow field from its spatial gradient
// (rotated 90° — geostrophic approximation) so particles "feel" like currents.

const { useEffect, useRef, useMemo } = React;

// ---------- SSH field synthesis ----------
// Generate a plausible Gulf-of-Mexico-shaped SSH grid as a function of time.
// Real app would sample NetCDF; here we layer moving Gaussian "eddies" + a
// Loop Current path that expands/sheds. Output: cols x rows in [-1, 1].

const COLS = 96, ROWS = 56;

// Gulf coastline mask — crude but evocative. 1 = water, 0 = land.
// Derived from a low-res silhouette so map has a recognizable shape.
function gulfMask(x, y) {
  // x,y in [0,1]. x=0 west (TX), x=1 east (FL), y=0 south (Yucatan), y=1 north coast.
  // Land: Mexico south coast, Texas/Louisiana north, Florida east.
  const nx = x * 2 - 1, ny = y * 2 - 1;
  // Main basin ellipse
  const basin = (nx*nx)/0.92 + (ny*ny)/0.78;
  if (basin > 1.02) return 0;
  // Florida peninsula cutout (top right)
  if (x > 0.82 && y > 0.55 && !(x > 0.93 && y < 0.72)) return 0;
  // Yucatan peninsula (bottom right-ish)
  if (x > 0.62 && x < 0.78 && y < 0.22) return 0;
  // Texas/Louisiana coast curve
  const coast = 0.82 + 0.08*Math.sin(x*Math.PI*1.6);
  if (y > coast) return 0;
  // Cuba sliver (bottom right)
  if (x > 0.78 && x < 0.98 && y > 0.18 && y < 0.30) return 0;
  // Mexico south coast
  if (y < 0.08 + 0.06*Math.sin(x*3.2)) return 0;
  return 1;
}

// Smooth noise
function hash(i, j, s) {
  const n = Math.sin(i*127.1 + j*311.7 + s*74.7) * 43758.5453;
  return n - Math.floor(n);
}
function vnoise(x, y, s) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const u = xf*xf*(3-2*xf), v = yf*yf*(3-2*yf);
  const a = hash(xi, yi, s), b = hash(xi+1, yi, s);
  const c = hash(xi, yi+1, s), d = hash(xi+1, yi+1, s);
  return (a*(1-u)+b*u)*(1-v) + (c*(1-u)+d*u)*v;
}

// Compute SSH value at normalized (x,y) and time t (0..1 over 40 years)
// NOTE: field is computed everywhere so hurricane signatures visually cross
// coastlines. Over land the value is damped (storms weaken after landfall).
function sshAt(x, y, t, anomaly = 0) {
  const onWater = gulfMask(x, y);
  // Baseline: slightly higher in center
  let v = 0.0;
  // Loop Current: a meandering ridge from Yucatan up & east toward FL straits
  // Animate expansion: the northward extent grows then sheds
  const lcPhase = (t * 12) % 1;           // cycles
  const lcReach = 0.55 + 0.25*Math.sin(t*18.8);
  // Path parameterized 0..1 from yucatan (0.7,0.1) up through basin
  for (let s = 0; s <= 1; s += 0.05) {
    const px = 0.72 - 0.18*Math.sin(s*Math.PI*0.6) + 0.06*Math.sin(s*6 + t*6);
    const py = 0.12 + lcReach*s;
    const d2 = (x-px)*(x-px) + (y-py)*(y-py);
    v += 0.55 * Math.exp(-d2/0.008);
  }
  // Shed eddies — a few moving blobs
  const eddies = [
    [0.55, 0.45, 0.6, 0.9],
    [0.40, 0.55, 0.5, 1.2],
    [0.30, 0.42, 0.45, 0.7],
    [0.62, 0.35, 0.55, 1.5],
    [0.48, 0.30, 0.4, 1.1],
  ];
  for (const [cx, cy, amp, spd] of eddies) {
    const ex = cx + 0.08*Math.sin(t*spd*6.28 + cx*10);
    const ey = cy + 0.05*Math.cos(t*spd*6.28 + cy*10);
    const d2 = (x-ex)*(x-ex) + (y-ey)*(y-ey);
    v += amp * Math.exp(-d2/0.012) * (0.7 + 0.3*Math.sin(t*spd*12));
  }
  // Negative eddies (cold)
  const cold = [[0.38, 0.38, -0.5, 0.8],[0.52, 0.58, -0.4, 1.0]];
  for (const [cx, cy, amp, spd] of cold) {
    const ex = cx + 0.06*Math.sin(t*spd*6 + cx*7);
    const ey = cy + 0.06*Math.cos(t*spd*6 + cy*7);
    const d2 = (x-ex)*(x-ex) + (y-ey)*(y-ey);
    v += amp * Math.exp(-d2/0.015);
  }
  // Fine texture
  v += (vnoise(x*9 + t*3, y*9, 3) - 0.5) * 0.25;
  v += (vnoise(x*22, y*22, 7) - 0.5) * 0.12;
  // Temperature anomaly amplifies positive SSH
  v = v * (1 + anomaly * 0.55);
  // Land damping — storms lose ocean heat input on landfall, so we fade the
  // SSH signature there to ~45% of its oceanic value. Cold anomalies (<0)
  // are pushed further toward neutral so we don't paint land with navy.
  if (!onWater) {
    v = v > 0 ? v * 0.45 : v * 0.2;
  }
  return Math.max(-1, Math.min(1.2, v));
}

// Build full grid
function buildGrid(t, anomaly) {
  const g = new Float32Array(COLS * ROWS);
  for (let j = 0; j < ROWS; j++) {
    for (let i = 0; i < COLS; i++) {
      const x = i / (COLS-1), y = 1 - j / (ROWS-1);
      g[j*COLS + i] = sshAt(x, y, t, anomaly);
    }
  }
  return g;
}

// SSH → RGBA
// -1 deep blue → 0 neutral teal → +1 bright red
function sshColor(v) {
  if (isNaN(v)) return [0,0,0,0];
  // clamp
  const x = Math.max(-1, Math.min(1.2, v));
  let r, g, b;
  if (x < 0) {
    const k = Math.max(0, Math.min(1, -x));
    // deep navy -> teal
    r = 6 + (20 - 6) * (1-k);
    g = 40 + (120 - 40) * (1-k);
    b = 120 + (170 - 120) * (1-k);
    // deepest: very dark
    r *= (1 - 0.6*k); g *= (1 - 0.4*k); b *= (1 - 0.2*k);
  } else {
    const k = Math.max(0, Math.min(1, x));
    // neutral teal -> amber -> crimson
    if (k < 0.5) {
      const u = k/0.5;
      r = 20 + (220 - 20)*u;
      g = 120 + (170 - 120)*u;
      b = 170 + (90  - 170)*u;
    } else {
      const u = (k-0.5)/0.5;
      r = 220 + (255 - 220)*u;
      g = 170 + (60  - 170)*u;
      b = 90  + (100 - 90)*u;
    }
    // Risk zone: SSH > 0.75 (≈ > 17 cm) — push toward hot magenta-red
    if (x > 0.75) {
      const u = Math.min(1, (x - 0.75)/0.4);
      r = r*(1-u) + 255*u;
      g = g*(1-u) + 40*u;
      b = b*(1-u) + 110*u;
    }
  }
  return [r|0, g|0, b|0, 255];
}

// Bilinear sample of grid
function sample(grid, x, y) {
  if (x < 0 || x > 1 || y < 0 || y > 1) return NaN;
  const fx = x*(COLS-1), fy = (1-y)*(ROWS-1);
  const i0 = Math.floor(fx), j0 = Math.floor(fy);
  const i1 = Math.min(COLS-1, i0+1), j1 = Math.min(ROWS-1, j0+1);
  const tx = fx-i0, ty = fy-j0;
  const a = grid[j0*COLS+i0], b = grid[j0*COLS+i1];
  const c = grid[j1*COLS+i0], d = grid[j1*COLS+i1];
  if (isNaN(a)||isNaN(b)||isNaN(c)||isNaN(d)) return NaN;
  return (a*(1-tx)+b*tx)*(1-ty) + (c*(1-tx)+d*tx)*ty;
}

// Flow field from SSH gradient, rotated 90° (geostrophic-ish)
function flowAt(grid, x, y) {
  const h = 0.02;
  const sL = sample(grid, x-h, y), sR = sample(grid, x+h, y);
  const sD = sample(grid, x, y-h), sU = sample(grid, x, y+h);
  if (isNaN(sL)||isNaN(sR)||isNaN(sD)||isNaN(sU)) return [0, 0];
  const dx = (sR - sL) / (2*h);
  const dy = (sU - sD) / (2*h);
  // rotate 90° CCW → geostrophic
  return [-dy, dx];
}

// ---------- Eddy detection (local maxima above threshold) ----------
function detectEddies(grid, thresh = 0.55) {
  const out = [];
  const seen = new Uint8Array(COLS*ROWS);
  for (let j = 2; j < ROWS-2; j++) {
    for (let i = 2; i < COLS-2; i++) {
      const v = grid[j*COLS+i];
      if (isNaN(v) || v < thresh) continue;
      // Reject peaks that fall on land (v is damped there so it rarely crosses
      // thresh, but guard explicitly in case a coastal cell just barely does).
      const x = i/(COLS-1), y = 1 - j/(ROWS-1);
      if (!gulfMask(x, y)) continue;
      let peak = true;
      for (let dj=-1; dj<=1 && peak; dj++)
        for (let di=-1; di<=1 && peak; di++)
          if ((di||dj) && grid[(j+dj)*COLS+(i+di)] > v) peak = false;
      if (peak && !seen[j*COLS+i]) {
        // radius: expand while >thresh*0.75
        let r = 1;
        while (r < 8) {
          let ok = true;
          for (let a=0; a<8 && ok; a++) {
            const ang = (a/8)*Math.PI*2;
            const ii = (i + Math.cos(ang)*r)|0;
            const jj = (j + Math.sin(ang)*r)|0;
            if (ii<0||ii>=COLS||jj<0||jj>=ROWS) { ok=false; break; }
            const vv = grid[jj*COLS+ii];
            if (isNaN(vv) || vv < thresh*0.6) ok = false;
          }
          if (!ok) break;
          r++;
        }
        out.push({
          x: i/(COLS-1), y: 1 - j/(ROWS-1),
          r: r/COLS,
          strength: v,
        });
        // suppress neighborhood
        for (let dj=-3; dj<=3; dj++)
          for (let di=-3; di<=3; di++) {
            const jj=j+dj, ii=i+di;
            if (jj>=0&&jj<ROWS&&ii>=0&&ii<COLS) seen[jj*COLS+ii] = 1;
          }
      }
    }
  }
  return out.sort((a,b)=>b.strength-a.strength).slice(0, 8);
}

// ---------- React component ----------
function GulfMap({
  t,                   // time 0..1 (40 yrs)
  anomaly,             // temp anomaly 0..3
  layers,              // { eddies, loopCurrent, riskZones, predictions }
  density,             // particle density 0..1.5
  glow,                // glow strength 0..1.5
  showGrid,
  onEddyCount,
  accent,
  mode,
}) {
  const heatRef = useRef(null);
  const particleRef = useRef(null);
  const overlayRef = useRef(null);
  const stateRef = useRef({ grid: null, particles: [], lastT: t, anomaly: anomaly });

  // Build grid reactively
  const grid = useMemo(() => buildGrid(t, anomaly), [t, anomaly]);
  const eddies = useMemo(() => layers.eddies ? detectEddies(grid) : [], [grid, layers.eddies]);

  useEffect(() => {
    onEddyCount && onEddyCount(detectEddies(grid).length);
  }, [grid]);

  // Paint heatmap whenever grid changes
  useEffect(() => {
    const c = heatRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    const w = c.width, h = c.height;
    // Draw low-res grid to offscreen, then blur via filter
    const off = document.createElement('canvas');
    off.width = COLS; off.height = ROWS;
    const octx = off.getContext('2d');
    const img = octx.createImageData(COLS, ROWS);
    for (let j = 0; j < ROWS; j++) {
      for (let i = 0; i < COLS; i++) {
        const v = grid[j*COLS+i];
        const [r,g,b,a] = sshColor(v);
        const p = (j*COLS+i)*4;
        img.data[p]=r; img.data[p+1]=g; img.data[p+2]=b; img.data[p+3]=a;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.filter = `blur(${2 + glow*1.5}px) saturate(1.15)`;
    ctx.drawImage(off, 0, 0, w, h);
    ctx.filter = 'none';

    // glow pass
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = 0.22 * glow;
    ctx.filter = `blur(${14 + glow*10}px) saturate(1.6)`;
    ctx.drawImage(off, 0, 0, w, h);
    ctx.filter = 'none';
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';

    // Land/mask — paint land atop as deep void
    // Not needed: mask returns NaN → transparent. Instead paint coastline.
  }, [grid, glow]);

  // Paint overlays (eddies, loop current, risk zones, labels)
  useEffect(() => {
    const c = overlayRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    const w = c.width, h = c.height;
    ctx.clearRect(0, 0, w, h);

    // Risk zones: contour SSH > 0.75
    if (layers.riskZones) {
      ctx.save();
      for (let j = 0; j < ROWS; j++) {
        for (let i = 0; i < COLS; i++) {
          const v = grid[j*COLS+i];
          if (isNaN(v) || v < 0.75) continue;
          const x = (i/(COLS-1))*w;
          const y = (j/(ROWS-1))*h;
          const cellW = w/COLS, cellH = h/ROWS;
          ctx.fillStyle = `rgba(255, 60, 110, ${0.08 + (v-0.75)*0.4})`;
          ctx.fillRect(x - cellW*0.5, y - cellH*0.5, cellW*1.5, cellH*1.5);
        }
      }
      ctx.restore();
    }

    // Loop Current: trace high ridge from yucatan
    if (layers.loopCurrent) {
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 200, 120, 0.85)';
      ctx.lineWidth = 2.2;
      ctx.shadowBlur = 18;
      ctx.shadowColor = 'rgba(255, 180, 80, 0.8)';
      ctx.beginPath();
      const lcReach = 0.55 + 0.25*Math.sin(t*18.8);
      for (let s = 0; s <= 1; s += 0.02) {
        const px = 0.72 - 0.18*Math.sin(s*Math.PI*0.6) + 0.06*Math.sin(s*6 + t*6);
        const py = 0.12 + lcReach*s;
        const X = px*w, Y = (1-py)*h;
        if (s === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      }
      ctx.stroke();
      ctx.restore();

      // label — bottom-left, out of the way
      ctx.font = '500 10px JetBrains Mono';
      ctx.fillStyle = 'rgba(255, 200, 140, 0.9)';
      ctx.fillText('LOOP CURRENT', 16, h - 92);
      ctx.fillStyle = 'rgba(255, 200, 140, 0.5)';
      ctx.fillText(`EXT ${(lcReach*100|0)}% · shedding τ+${((1-lcReach)*40|0)}d`, 16, h - 78);
    }

    // Eddies
    if (layers.eddies) {
      ctx.save();
      for (const e of eddies) {
        const X = e.x*w, Y = (1-e.y)*h;
        const R = e.r*w*2.4 + 14;
        const grd = ctx.createRadialGradient(X, Y, R*0.3, X, Y, R);
        grd.addColorStop(0, 'rgba(120, 240, 255, 0.0)');
        grd.addColorStop(0.7, `rgba(120, 240, 255, ${0.15*e.strength})`);
        grd.addColorStop(1, 'rgba(120, 240, 255, 0)');
        ctx.fillStyle = grd;
        ctx.beginPath(); ctx.arc(X, Y, R, 0, Math.PI*2); ctx.fill();

        ctx.strokeStyle = `rgba(180, 245, 255, ${0.55*e.strength})`;
        ctx.lineWidth = 1.1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.arc(X, Y, R*0.72, 0, Math.PI*2); ctx.stroke();
        ctx.setLineDash([]);

        // ticker — only the strongest 3 get labels, rest just ring
        if (eddies.indexOf(e) < 3) {
          ctx.font = '500 9px JetBrains Mono';
          ctx.fillStyle = 'rgba(180, 240, 255, 0.75)';
          ctx.fillText(`E·${(e.strength*17+8).toFixed(0)}cm`, X + R*0.8, Y + 3);
        }
      }
      ctx.restore();
    }

    // Risk labels — cluster peaks, dedupe nearby, cap at 2, push labels to map edges
    if (layers.riskZones) {
      const peaks = [];
      for (let j = 2; j < ROWS-2; j++) {
        for (let i = 2; i < COLS-2; i++) {
          const v = grid[j*COLS+i];
          if (isNaN(v) || v < 0.88) continue;
          let isPeak = true;
          for (let dj=-1; dj<=1 && isPeak; dj++)
            for (let di=-1; di<=1 && isPeak; di++)
              if ((di||dj) && (grid[(j+dj)*COLS+(i+di)]||0) > v) isPeak = false;
          if (isPeak) peaks.push({x:i/(COLS-1), y:1-j/(ROWS-1), v, px:i*(w/COLS), py:j*(h/ROWS)});
        }
      }
      peaks.sort((a,b)=>b.v-a.v);
      // dedupe within 120px
      const kept = [];
      for (const p of peaks) {
        if (kept.every(k => Math.hypot(k.px-p.px, k.py-p.py) > 120)) kept.push(p);
        if (kept.length >= 2) break;
      }
      ctx.save();
      ctx.font = '500 10px JetBrains Mono';
      kept.forEach((r, idx) => {
        const X = r.px, Y = r.py;
        // right-side labels always; stagger vertically
        const labelX = w - 166;
        const labelY = 36 + idx * 44;
        // leader
        ctx.strokeStyle = 'rgba(255, 100, 140, 0.7)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(X, Y);
        ctx.lineTo(labelX - 6, labelY + 6);
        ctx.stroke();
        ctx.setLineDash([]);
        // target dot
        ctx.fillStyle = 'rgba(255, 80, 120, 0.95)';
        ctx.beginPath(); ctx.arc(X, Y, 3, 0, Math.PI*2); ctx.fill();
        ctx.strokeStyle = 'rgba(255, 80, 120, 0.5)';
        ctx.beginPath(); ctx.arc(X, Y, 8, 0, Math.PI*2); ctx.stroke();
        // label background
        ctx.fillStyle = 'rgba(20, 8, 18, 0.85)';
        ctx.fillRect(labelX - 4, labelY - 2, 158, 28);
        ctx.strokeStyle = 'rgba(255, 100, 140, 0.5)';
        ctx.strokeRect(labelX - 4, labelY - 2, 158, 28);
        // text
        ctx.fillStyle = '#ff6a8a';
        ctx.fillText('RAPID INTENSIFICATION', labelX, labelY + 9);
        ctx.fillStyle = 'rgba(255, 180, 200, 0.85)';
        ctx.fillText(`SSH +${(r.v*17).toFixed(1)} cm  · ${(18+r.y*12).toFixed(1)}°N`, labelX, labelY + 21);
      });
      ctx.restore();
    }

    // Grid overlay
    if (showGrid) {
      ctx.save();
      ctx.strokeStyle = 'rgba(120, 180, 240, 0.05)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 10; i++) {
        const x = (i/10)*w;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let j = 0; j <= 6; j++) {
        const y = (j/6)*h;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      // lat/lon ticks
      ctx.fillStyle = 'rgba(120, 180, 240, 0.4)';
      ctx.font = '9px JetBrains Mono';
      const lats = ['18°N','22°N','26°N','30°N'];
      const lons = ['98°W','92°W','86°W','80°W'];
      lats.forEach((s,i)=> ctx.fillText(s, 4, h - (i/3)*(h*0.9) - 6));
      lons.forEach((s,i)=> ctx.fillText(s, (i/3)*(w*0.95)+12, 12));
      ctx.restore();
    }
  }, [grid, eddies, layers, t, showGrid]);

  // Particle system
  useEffect(() => {
    const c = particleRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    const w = c.width, h = c.height;
    // fade trail
    let raf;
    const N = Math.floor(1800 * density);
    const parts = stateRef.current.particles;
    // resize pool
    while (parts.length < N) parts.push(spawn());
    if (parts.length > N) parts.length = N;

    function spawn() {
      // Prefer spawning in water so the flow field has meaningful bias; but
      // particles are now allowed to drift over land with reduced opacity
      // (they just fade rather than die).
      let x, y, tries = 0;
      do {
        x = Math.random(); y = Math.random();
        tries++;
      } while (!gulfMask(x, y) && tries < 20);
      return { x, y, life: Math.random()*60|0, maxLife: 40 + (Math.random()*80)|0 };
    }

    function step() {
      // trail fade
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = 'rgba(3, 7, 14, 0.12)';
      ctx.fillRect(0, 0, w, h);
      ctx.globalCompositeOperation = 'lighter';

      const g = stateRef.current.grid;
      if (!g) return (raf = requestAnimationFrame(step));

      for (let p = 0; p < parts.length; p++) {
        const P = parts[p];
        const [fx, fy] = flowAt(g, P.x, P.y);
        const spd = Math.hypot(fx, fy);
        const nx = P.x + fx * 0.003;
        const ny = P.y + fy * 0.003;
        const v = sample(g, P.x, P.y);

        // Respawn only when particle exits the canvas or runs out of life.
        // Land is no longer a hard kill zone — storms cross it while fading.
        if (nx < 0 || nx > 1 || ny < 0 || ny > 1 ||
            P.life++ > P.maxLife || isNaN(v)) {
          Object.assign(P, spawn());
          continue;
        }

        const overLand = !gulfMask(P.x, P.y);

        // color by local SSH
        let r, gC, b;
        if (v < 0) { r=80; gC=200; b=255; }
        else if (v < 0.5) { r=120; gC=240; b=220; }
        else if (v < 0.8) { r=255; gC=210; b=130; }
        else { r=255; gC=90; b=130; }

        // Over land: particle trails read as ghosted/fading rather than
        // disappearing — physically matches a weakening storm after landfall.
        const landFade = overLand ? 0.35 : 1.0;
        const alpha = Math.min(1, spd*4) * (0.5 + 0.5*Math.sin(P.life/P.maxLife*Math.PI)) * landFade;
        ctx.strokeStyle = `rgba(${r},${gC},${b},${alpha*0.9})`;
        ctx.lineWidth = 0.9 + (v > 0.6 ? 0.6 : 0);
        ctx.beginPath();
        ctx.moveTo(P.x*w, (1-P.y)*h);
        ctx.lineTo(nx*w, (1-ny)*h);
        ctx.stroke();
        P.x = nx; P.y = ny;
      }
      raf = requestAnimationFrame(step);
    }
    stateRef.current.particles = parts;
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [density]);

  // update grid ref
  useEffect(() => { stateRef.current.grid = grid; }, [grid]);

  return (
    <div className="gw-map-wrap">
      <canvas ref={heatRef} width={960} height={560} className="gw-canvas gw-heat" />
      <canvas ref={particleRef} width={960} height={560} className="gw-canvas gw-parts" />
      <canvas ref={overlayRef} width={960} height={560} className="gw-canvas gw-overlay" />
      {/* Coastline accent using CSS mask (decorative) */}
      <svg className="gw-coast" viewBox="0 0 960 560" preserveAspectRatio="none">
        <defs>
          <filter id="coastGlow"><feGaussianBlur stdDeviation="0.8"/></filter>
        </defs>
        {/* Texas/LA arc */}
        <path d="M 20,100 Q 220,60 440,90 T 820,110" stroke="rgba(180,220,255,0.35)" strokeWidth="1" fill="none" filter="url(#coastGlow)"/>
        {/* Florida */}
        <path d="M 820,110 Q 870,180 870,260 Q 860,320 830,360" stroke="rgba(180,220,255,0.35)" strokeWidth="1" fill="none"/>
        {/* Cuba */}
        <path d="M 760,440 Q 820,430 880,450" stroke="rgba(180,220,255,0.3)" strokeWidth="1" fill="none"/>
        {/* Yucatan / Mexico */}
        <path d="M 660,530 Q 640,480 650,440 Q 600,420 520,480 Q 380,520 220,500 Q 80,490 20,460" stroke="rgba(180,220,255,0.35)" strokeWidth="1" fill="none"/>
      </svg>
    </div>
  );
}

window.GulfMap = GulfMap;
window.buildGrid = buildGrid;
window.detectEddies = detectEddies;
window.sshAt = sshAt;
window.COLS = COLS;
window.ROWS = ROWS;
