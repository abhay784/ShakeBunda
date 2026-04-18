'use client';

import React, { useEffect, useRef, useMemo, useState } from 'react';
import {
  COLS,
  ROWS,
  gulfMask,
  buildGrid,
  sshColor,
  sample,
  flowAt,
  detectEddies,
  type Eddy,
} from '@/lib/ocean/ssh';

interface GulfMapProps {
  t: number; // time 0..1 (40 yrs)
  anomaly: number; // temp anomaly 0..3
  layers: {
    eddies: boolean;
    loopCurrent: boolean;
    riskZones: boolean;
    predictions: boolean;
  };
  density: number; // particle density 0..1.5
  glow: number; // glow strength 0..1.5
  showGrid: boolean;
  onEddyCount?: (count: number) => void;
  mode: 'historical' | 'predicted';
  riGrid?: number[][]; // from Databricks API; overrides synthetic grid if present
}

export function GulfMap({
  t,
  anomaly,
  layers,
  density,
  glow,
  showGrid,
  onEddyCount,
  mode,
  riGrid,
}: GulfMapProps) {
  const heatRef = useRef<HTMLCanvasElement>(null);
  const particleRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef<{
    grid: Float32Array | null;
    particles: Array<{ x: number; y: number; life: number; maxLife: number }>;
  }>({ grid: null, particles: [] });

  // Build grid or use Databricks result
  const grid = useMemo(() => {
    if (riGrid && riGrid.length > 0) {
      // Convert API grid (2D array) to our format
      const g = new Float32Array(COLS * ROWS);
      for (let j = 0; j < Math.min(ROWS, riGrid.length); j++) {
        for (let i = 0; i < Math.min(COLS, riGrid[j].length); i++) {
          // Normalize [0,1] to [-1, 1.2] range for visualization
          g[j * COLS + i] = (riGrid[j][i] - 0.5) * 2.4;
        }
      }
      return g;
    }
    return buildGrid(t, anomaly);
  }, [t, anomaly, riGrid]);

  const eddies = useMemo(
    () => (layers.eddies ? detectEddies(grid) : []),
    [grid, layers.eddies],
  );

  useEffect(() => {
    onEddyCount?.(detectEddies(grid).length);
  }, [grid, onEddyCount]);

  // Heatmap rendering
  useEffect(() => {
    const c = heatRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const w = c.width;
    const h = c.height;

    // Low-res grid → blur for smoothness
    const off = document.createElement('canvas');
    off.width = COLS;
    off.height = ROWS;
    const octx = off.getContext('2d');
    if (!octx) return;

    const img = octx.createImageData(COLS, ROWS);
    for (let j = 0; j < ROWS; j++) {
      for (let i = 0; i < COLS; i++) {
        const v = grid[j * COLS + i];
        const [r, g, b, a] = sshColor(v);
        const p = (j * COLS + i) * 4;
        img.data[p] = r;
        img.data[p + 1] = g;
        img.data[p + 2] = b;
        img.data[p + 3] = a;
      }
    }
    octx.putImageData(img, 0, 0);

    ctx.clearRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.filter = `blur(${2 + glow * 1.5}px) saturate(1.15)`;
    ctx.drawImage(off, 0, 0, w, h);
    ctx.filter = 'none';

    // Glow pass
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = 0.22 * glow;
    ctx.filter = `blur(${14 + glow * 10}px) saturate(1.6)`;
    ctx.drawImage(off, 0, 0, w, h);
    ctx.filter = 'none';
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
  }, [grid, glow]);

  // Overlay rendering (eddies, loop current, risk zones)
  useEffect(() => {
    const c = overlayRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const w = c.width;
    const h = c.height;

    ctx.clearRect(0, 0, w, h);

    // Risk zones: SSH > 0.75
    if (layers.riskZones) {
      for (let j = 0; j < ROWS; j++) {
        for (let i = 0; i < COLS; i++) {
          const v = grid[j * COLS + i];
          if (isNaN(v) || v < 0.75) continue;
          const x = (i / (COLS - 1)) * w;
          const y = (j / (ROWS - 1)) * h;
          const cellW = w / COLS;
          const cellH = h / ROWS;
          ctx.fillStyle = `rgba(255, 60, 110, ${0.08 + (v - 0.75) * 0.4})`;
          ctx.fillRect(x - cellW * 0.5, y - cellH * 0.5, cellW * 1.5, cellH * 1.5);
        }
      }
    }

    // Loop Current trace
    if (layers.loopCurrent) {
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 200, 120, 0.85)';
      ctx.lineWidth = 2.2;
      ctx.shadowBlur = 18;
      ctx.shadowColor = 'rgba(255, 180, 80, 0.8)';
      ctx.beginPath();
      const lcReach = 0.55 + 0.25 * Math.sin(t * 18.8);
      for (let s = 0; s <= 1; s += 0.02) {
        const px = 0.72 - 0.18 * Math.sin(s * Math.PI * 0.6) + 0.06 * Math.sin(s * 6 + t * 6);
        const py = 0.12 + lcReach * s;
        const X = px * w;
        const Y = (1 - py) * h;
        if (s === 0) ctx.moveTo(X, Y);
        else ctx.lineTo(X, Y);
      }
      ctx.stroke();

      ctx.font = '500 10px JetBrains Mono';
      ctx.fillStyle = 'rgba(255, 200, 140, 0.9)';
      ctx.fillText('LOOP CURRENT', 16, h - 92);
      ctx.fillStyle = 'rgba(255, 200, 140, 0.5)';
      ctx.fillText(
        `EXT ${(lcReach * 100) | 0}% · shedding τ+${((1 - lcReach) * 40) | 0}d`,
        16,
        h - 78,
      );
      ctx.restore();
    }

    // Eddies
    if (layers.eddies) {
      for (const e of eddies) {
        const X = e.x * w;
        const Y = (1 - e.y) * h;
        const R = e.r * w * 2.4 + 14;
        const grd = ctx.createRadialGradient(X, Y, R * 0.3, X, Y, R);
        grd.addColorStop(0, 'rgba(120, 240, 255, 0.0)');
        grd.addColorStop(0.7, `rgba(120, 240, 255, ${0.15 * e.strength})`);
        grd.addColorStop(1, 'rgba(120, 240, 255, 0)');
        ctx.fillStyle = grd;
        ctx.beginPath();
        ctx.arc(X, Y, R, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = `rgba(180, 245, 255, ${0.55 * e.strength})`;
        ctx.lineWidth = 1.1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.arc(X, Y, R * 0.72, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        if (eddies.indexOf(e) < 3) {
          ctx.font = '500 9px JetBrains Mono';
          ctx.fillStyle = 'rgba(180, 240, 255, 0.75)';
          ctx.fillText(`E·${(e.strength * 17 + 8).toFixed(0)}cm`, X + R * 0.8, Y + 3);
        }
      }
    }

    // Risk labels
    if (layers.riskZones) {
      const peaks: Array<{ x: number; y: number; v: number; px: number; py: number }> = [];
      for (let j = 2; j < ROWS - 2; j++) {
        for (let i = 2; i < COLS - 2; i++) {
          const v = grid[j * COLS + i];
          if (isNaN(v) || v < 0.88) continue;
          let isPeak = true;
          for (let dj = -1; dj <= 1 && isPeak; dj++) {
            for (let di = -1; di <= 1 && isPeak; di++) {
              if ((di || dj) && (grid[(j + dj) * COLS + (i + di)] || 0) > v) {
                isPeak = false;
              }
            }
          }
          if (isPeak) {
            peaks.push({
              x: i / (COLS - 1),
              y: 1 - j / (ROWS - 1),
              v,
              px: i * (w / COLS),
              py: j * (h / ROWS),
            });
          }
        }
      }

      peaks.sort((a, b) => b.v - a.v);
      const kept: typeof peaks = [];
      for (const p of peaks) {
        if (kept.every((k) => Math.hypot(k.px - p.px, k.py - p.py) > 120)) {
          kept.push(p);
        }
        if (kept.length >= 2) break;
      }

      ctx.font = '500 10px JetBrains Mono';
      kept.forEach((r, idx) => {
        const X = r.px;
        const Y = r.py;
        const labelX = w - 166;
        const labelY = 36 + idx * 44;

        ctx.strokeStyle = 'rgba(255, 100, 140, 0.7)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(X, Y);
        ctx.lineTo(labelX - 6, labelY + 6);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = 'rgba(255, 80, 120, 0.95)';
        ctx.beginPath();
        ctx.arc(X, Y, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 80, 120, 0.5)';
        ctx.beginPath();
        ctx.arc(X, Y, 8, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = 'rgba(20, 8, 18, 0.85)';
        ctx.fillRect(labelX - 4, labelY - 2, 158, 28);
        ctx.strokeStyle = 'rgba(255, 100, 140, 0.5)';
        ctx.strokeRect(labelX - 4, labelY - 2, 158, 28);

        ctx.fillStyle = '#ff6a8a';
        ctx.fillText('RAPID INTENSIFICATION', labelX, labelY + 9);
        ctx.fillStyle = 'rgba(255, 180, 200, 0.85)';
        ctx.fillText(
          `SSH +${(r.v * 17).toFixed(1)} cm  · ${(18 + r.y * 12).toFixed(1)}°N`,
          labelX,
          labelY + 21,
        );
      });
    }

    // Grid overlay
    if (showGrid) {
      ctx.strokeStyle = 'rgba(120, 180, 240, 0.05)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 10; i++) {
        const x = (i / 10) * w;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let j = 0; j <= 6; j++) {
        const y = (j / 6) * h;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      ctx.fillStyle = 'rgba(120, 180, 240, 0.4)';
      ctx.font = '9px JetBrains Mono';
      const lats = ['18°N', '22°N', '26°N', '30°N'];
      const lons = ['98°W', '92°W', '86°W', '80°W'];
      lats.forEach((s, i) => ctx.fillText(s, 4, h - (i / 3) * (h * 0.9) - 6));
      lons.forEach((s, i) => ctx.fillText(s, (i / 3) * (w * 0.95) + 12, 12));
    }
  }, [grid, eddies, layers, t, showGrid]);

  // Particle system
  useEffect(() => {
    const c = particleRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const w = c.width;
    const h = c.height;

    const N = Math.floor(1800 * density);
    const parts = stateRef.current.particles;
    while (parts.length < N) parts.push(spawn());
    if (parts.length > N) parts.length = N;

    function spawn() {
      let x: number, y: number, tries = 0;
      do {
        x = Math.random();
        y = Math.random();
        tries++;
      } while (!gulfMask(x, y) && tries < 20);
      return {
        x,
        y,
        life: (Math.random() * 60) | 0,
        maxLife: 40 + ((Math.random() * 80) | 0),
      };
    }

    let raf: number;
    function step() {
      if (!ctx) return;
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = 'rgba(3, 7, 14, 0.12)';
      ctx.fillRect(0, 0, w, h);
      ctx.globalCompositeOperation = 'lighter';

      const g = stateRef.current.grid;
      if (!g) {
        raf = requestAnimationFrame(step);
        return;
      }

      for (let p = 0; p < parts.length; p++) {
        const P = parts[p];
        const [fx, fy] = flowAt(g, P.x, P.y);
        const nx = P.x + fx * 0.003;
        const ny = P.y + fy * 0.003;
        const v = sample(g, P.x, P.y);

        if (!gulfMask(P.x, P.y) || P.life++ > P.maxLife || isNaN(v)) {
          Object.assign(P, spawn());
          continue;
        }

        let r: number, gC: number, b: number;
        if (v < 0) {
          r = 80;
          gC = 200;
          b = 255;
        } else if (v < 0.5) {
          r = 120;
          gC = 240;
          b = 220;
        } else if (v < 0.8) {
          r = 255;
          gC = 210;
          b = 130;
        } else {
          r = 255;
          gC = 90;
          b = 130;
        }

        const alpha = Math.min(1, Math.abs(fx) + Math.abs(fy) * 4) * (0.5 + 0.5 * Math.sin((P.life / P.maxLife) * Math.PI));
        ctx.strokeStyle = `rgba(${r},${gC},${b},${alpha * 0.9})`;
        ctx.lineWidth = 0.9 + (v > 0.6 ? 0.6 : 0);
        ctx.beginPath();
        ctx.moveTo(P.x * w, (1 - P.y) * h);
        ctx.lineTo(nx * w, (1 - ny) * h);
        ctx.stroke();
        P.x = nx;
        P.y = ny;
      }
      raf = requestAnimationFrame(step);
    }
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [density]);

  useEffect(() => {
    stateRef.current.grid = grid;
  }, [grid]);

  return (
    <div className="gw-map-wrap">
      <canvas ref={heatRef} width={960} height={560} className="gw-canvas gw-heat" />
      <canvas ref={particleRef} width={960} height={560} className="gw-canvas gw-parts" />
      <canvas ref={overlayRef} width={960} height={560} className="gw-canvas gw-overlay" />
      <svg className="gw-coast" viewBox="0 0 960 560" preserveAspectRatio="none">
        <defs>
          <filter id="coastGlow">
            <feGaussianBlur stdDeviation="0.8" />
          </filter>
        </defs>
        <path
          d="M 20,100 Q 220,60 440,90 T 820,110"
          stroke="rgba(180,220,255,0.35)"
          strokeWidth="1"
          fill="none"
          filter="url(#coastGlow)"
        />
        <path
          d="M 820,110 Q 870,180 870,260 Q 860,320 830,360"
          stroke="rgba(180,220,255,0.35)"
          strokeWidth="1"
          fill="none"
        />
        <path
          d="M 760,440 Q 820,430 880,450"
          stroke="rgba(180,220,255,0.3)"
          strokeWidth="1"
          fill="none"
        />
        <path
          d="M 660,530 Q 640,480 650,440 Q 600,420 520,480 Q 380,520 220,500 Q 80,490 20,460"
          stroke="rgba(180,220,255,0.35)"
          strokeWidth="1"
          fill="none"
        />
      </svg>
    </div>
  );
}
