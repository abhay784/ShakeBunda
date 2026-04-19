// Gulf of Mexico SSH field synthesis + visualization
// Real app would sample NetCDF; here we layer moving Gaussian eddies + a Loop
// Current path. Output: cols x rows in [-1, 1].

export const COLS = 96;
export const ROWS = 56;

// Gulf coastline mask — crude but evocative. 1 = water, 0 = land.
export function gulfMask(x: number, y: number): number {
  // x,y in [0,1]. x=0 west (TX), x=1 east (FL), y=0 south (Yucatan), y=1 north coast.
  const nx = x * 2 - 1;
  const ny = y * 2 - 1;

  // Main basin ellipse
  const basin = (nx * nx) / 0.92 + (ny * ny) / 0.78;
  if (basin > 1.02) return 0;

  // Florida peninsula cutout
  if (x > 0.82 && y > 0.55 && !(x > 0.93 && y < 0.72)) return 0;

  // Yucatan peninsula
  if (x > 0.62 && x < 0.78 && y < 0.22) return 0;

  // Texas/Louisiana coast
  const coast = 0.82 + 0.08 * Math.sin(x * Math.PI * 1.6);
  if (y > coast) return 0;

  // Cuba sliver
  if (x > 0.78 && x < 0.98 && y > 0.18 && y < 0.3) return 0;

  // Mexico south coast
  if (y < 0.08 + 0.06 * Math.sin(x * 3.2)) return 0;

  return 1;
}

// Smooth noise via value noise
function hash(i: number, j: number, s: number): number {
  const n = Math.sin(i * 127.1 + j * 311.7 + s * 74.7) * 43758.5453;
  return n - Math.floor(n);
}

function vnoise(x: number, y: number, s: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const a = hash(xi, yi, s);
  const b = hash(xi + 1, yi, s);
  const c = hash(xi, yi + 1, s);
  const d = hash(xi + 1, yi + 1, s);
  return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v;
}

// Compute SSH value at normalized (x,y) and time t (0..1 over 40 years)
export function sshAt(x: number, y: number, t: number, anomaly = 0): number {
  if (!gulfMask(x, y)) return NaN;

  let v = 0;

  // Loop Current: meandering ridge from Yucatan up & east toward FL straits
  const lcReach = 0.55 + 0.25 * Math.sin(t * 18.8);
  for (let s = 0; s <= 1; s += 0.05) {
    const px = 0.72 - 0.18 * Math.sin(s * Math.PI * 0.6) + 0.06 * Math.sin(s * 6 + t * 6);
    const py = 0.12 + lcReach * s;
    const d2 = (x - px) * (x - px) + (y - py) * (y - py);
    v += 0.55 * Math.exp(-d2 / 0.008);
  }

  // Shed eddies
  const eddies: [number, number, number, number][] = [
    [0.55, 0.45, 0.6, 0.9],
    [0.4, 0.55, 0.5, 1.2],
    [0.3, 0.42, 0.45, 0.7],
    [0.62, 0.35, 0.55, 1.5],
    [0.48, 0.3, 0.4, 1.1],
  ];

  for (const [cx, cy, amp, spd] of eddies) {
    const ex = cx + 0.08 * Math.sin(t * spd * 6.28 + cx * 10);
    const ey = cy + 0.05 * Math.cos(t * spd * 6.28 + cy * 10);
    const d2 = (x - ex) * (x - ex) + (y - ey) * (y - ey);
    v += amp * Math.exp(-d2 / 0.012) * (0.7 + 0.3 * Math.sin(t * spd * 12));
  }

  // Cold eddies
  const cold: [number, number, number, number][] = [
    [0.38, 0.38, -0.5, 0.8],
    [0.52, 0.58, -0.4, 1.0],
  ];

  for (const [cx, cy, amp, spd] of cold) {
    const ex = cx + 0.06 * Math.sin(t * spd * 6 + cx * 7);
    const ey = cy + 0.06 * Math.cos(t * spd * 6 + cy * 7);
    const d2 = (x - ex) * (x - ex) + (y - ey) * (y - ey);
    v += amp * Math.exp(-d2 / 0.015);
  }

  // Fine texture
  v += (vnoise(x * 9 + t * 3, y * 9, 3) - 0.5) * 0.25;
  v += (vnoise(x * 22, y * 22, 7) - 0.5) * 0.12;

  // Temperature anomaly amplifies positive SSH
  v = v * (1 + anomaly * 0.55);

  return Math.max(-1, Math.min(1.2, v));
}

// Build full grid
export function buildGrid(t: number, anomaly: number): Float32Array {
  const g = new Float32Array(COLS * ROWS);
  for (let j = 0; j < ROWS; j++) {
    for (let i = 0; i < COLS; i++) {
      const x = i / (COLS - 1);
      const y = 1 - j / (ROWS - 1);
      g[j * COLS + i] = sshAt(x, y, t, anomaly);
    }
  }
  return g;
}

// SSH → RGBA colormap
export function sshColor(v: number): [number, number, number, number] {
  if (isNaN(v)) return [0, 0, 0, 0];

  const x = Math.max(-1, Math.min(1.2, v));
  let r: number, g: number, b: number;

  if (x < 0) {
    const k = Math.max(0, Math.min(1, -x));
    r = (6 + (20 - 6) * (1 - k)) * (1 - 0.6 * k);
    g = (40 + (120 - 40) * (1 - k)) * (1 - 0.4 * k);
    b = (120 + (170 - 120) * (1 - k)) * (1 - 0.2 * k);
  } else {
    const k = Math.max(0, Math.min(1, x));
    if (k < 0.5) {
      const u = k / 0.5;
      r = 20 + (220 - 20) * u;
      g = 120 + (170 - 120) * u;
      b = 170 + (90 - 170) * u;
    } else {
      const u = (k - 0.5) / 0.5;
      r = 220 + (255 - 220) * u;
      g = 170 + (60 - 170) * u;
      b = 90 + (100 - 90) * u;
    }

    // Risk zone: SSH > 0.75 (≈ > 17 cm)
    if (x > 0.75) {
      const u = Math.min(1, (x - 0.75) / 0.4);
      r = r * (1 - u) + 255 * u;
      g = g * (1 - u) + 40 * u;
      b = b * (1 - u) + 110 * u;
    }
  }

  return [r | 0, g | 0, b | 0, 255];
}

// Bilinear sample of grid
export function sample(grid: Float32Array, x: number, y: number): number {
  if (x < 0 || x > 1 || y < 0 || y > 1) return NaN;
  const fx = x * (COLS - 1);
  const fy = (1 - y) * (ROWS - 1);
  const i0 = Math.floor(fx);
  const j0 = Math.floor(fy);
  const i1 = Math.min(COLS - 1, i0 + 1);
  const j1 = Math.min(ROWS - 1, j0 + 1);
  const tx = fx - i0;
  const ty = fy - j0;
  const a = grid[j0 * COLS + i0];
  const b = grid[j0 * COLS + i1];
  const c = grid[j1 * COLS + i0];
  const d = grid[j1 * COLS + i1];
  if (isNaN(a) || isNaN(b) || isNaN(c) || isNaN(d)) return NaN;
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

// Flow field from SSH gradient, rotated 90° (geostrophic-ish)
export function flowAt(
  grid: Float32Array,
  x: number,
  y: number,
): [number, number] {
  const h = 0.02;
  const sL = sample(grid, x - h, y);
  const sR = sample(grid, x + h, y);
  const sD = sample(grid, x, y - h);
  const sU = sample(grid, x, y + h);
  if (isNaN(sL) || isNaN(sR) || isNaN(sD) || isNaN(sU)) return [0, 0];
  const dx = (sR - sL) / (2 * h);
  const dy = (sU - sD) / (2 * h);
  return [-dy, dx];
}

// Eddy detection (local maxima above threshold)
export interface Eddy {
  x: number;
  y: number;
  r: number;
  strength: number;
}

export function detectEddies(grid: Float32Array, thresh = 0.68): Eddy[] {
  const out: Eddy[] = [];
  const seen = new Uint8Array(COLS * ROWS);

  for (let j = 2; j < ROWS - 2; j++) {
    for (let i = 2; i < COLS - 2; i++) {
      const v = grid[j * COLS + i];
      if (isNaN(v) || v < thresh) continue;

      let peak = true;
      for (let dj = -1; dj <= 1 && peak; dj++) {
        for (let di = -1; di <= 1 && peak; di++) {
          if ((di || dj) && grid[(j + dj) * COLS + (i + di)] > v) peak = false;
        }
      }

      if (peak && !seen[j * COLS + i]) {
        let r = 1;
        while (r < 8) {
          let ok = true;
          for (let a = 0; a < 8 && ok; a++) {
            const ang = (a / 8) * Math.PI * 2;
            const ii = (i + Math.cos(ang) * r) | 0;
            const jj = (j + Math.sin(ang) * r) | 0;
            if (ii < 0 || ii >= COLS || jj < 0 || jj >= ROWS) {
              ok = false;
              break;
            }
            const vv = grid[jj * COLS + ii];
            if (isNaN(vv) || vv < thresh * 0.6) ok = false;
          }
          if (!ok) break;
          r++;
        }

        out.push({
          x: i / (COLS - 1),
          y: 1 - j / (ROWS - 1),
          r: r / COLS,
          strength: v,
        });

        // Suppress neighborhood
        for (let dj = -3; dj <= 3; dj++) {
          for (let di = -3; di <= 3; di++) {
            const jj = j + dj;
            const ii = i + di;
            if (jj >= 0 && jj < ROWS && ii >= 0 && ii < COLS) {
              seen[jj * COLS + ii] = 1;
            }
          }
        }
      }
    }
  }

  return out.sort((a, b) => b.strength - a.strength);
}
