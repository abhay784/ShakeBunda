'use client';

import React, { useRef, useState } from 'react';

export const EVENTS = [
  { t: (1992 - 1985) / 40, label: 'ANDREW',  year: 1992, cat: 'Cat 5', note: 'FL landfall · bypassed warm Loop Current core' },
  { t: (1995 - 1985) / 40, label: 'OPAL',    year: 1995, cat: 'Cat 4', note: 'rapid intensification over warm eddy' },
  { t: (2005 - 1985) / 40, label: 'KATRINA', year: 2005, cat: 'Cat 5', note: 'crossed warm-core ring in central Gulf' },
  { t: (2005.7 - 1985) / 40, label: 'RITA',  year: 2005, cat: 'Cat 5', note: 'peak SSH anomaly +32 cm' },
  { t: (2017 - 1985) / 40, label: 'HARVEY',  year: 2017, cat: 'Cat 4', note: 'stalled over TX — +18 cm SSH' },
  { t: (2018.75 - 1985) / 40, label: 'MICHAEL', year: 2018, cat: 'Cat 5', note: 'intensified over Loop Current tongue' },
  { t: (2020.8 - 1985) / 40, label: 'LAURA', year: 2020, cat: 'Cat 4', note: 'warm shelf waters' },
  { t: (2021 - 1985) / 40, label: 'IDA',     year: 2021, cat: 'Cat 4', note: '65 mph → 150 mph in 24h — loop eddy' },
  { t: (2024.6 - 1985) / 40, label: 'MILTON', year: 2024, cat: 'Cat 5', note: 'record SSH +34 cm near Yucatán' },
];

const EVENT_ROWS = (() => {
  const sorted = [...EVENTS].sort((a, b) => a.t - b.t);
  const rows = new Map<string, number>();
  const lastRowTime = [-1, -1];
  sorted.forEach(e => {
    const row = (e.t - lastRowTime[0]) > (e.t - lastRowTime[1]) ? 0 : 1;
    rows.set(e.label + e.year, row);
    lastRowTime[row] = e.t;
  });
  return rows;
})();

interface TimelineProps {
  t: number;
  setT: (t: number) => void;
  playing: boolean;
}

export function Timeline({ t, setT, playing }: TimelineProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  function onTrack(e: React.MouseEvent) {
    const r = trackRef.current!.getBoundingClientRect();
    const nt = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    setT(nt);
  }

  const yearVal = 1985 + t * 40;
  const year = Math.floor(yearVal);
  const month = Math.floor((yearVal - year) * 12);
  const mon = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'][month];

  const years: number[] = [];
  for (let y = 1985; y <= 2025; y += 5) years.push(y);

  return (
    <div className="gw-timeline">
      <div className="gw-tl-head">
        <div className="gw-tl-date">
          <b>{mon} {year}</b>
          <em>day {Math.floor(((yearVal - year) * 365) + 1).toString().padStart(3, '0')} · {(t * 100).toFixed(1)}%</em>
        </div>
        <div className="gw-tl-events">
          {[EVENTS[2], EVENTS[4], EVENTS[7]].map(e => (
            <button key={e.label} className="gw-tl-jump" onClick={() => setT(e.t)}>
              {e.year} <b>{e.label}</b>
            </button>
          ))}
        </div>
      </div>

      <div className="gw-tl-track-wrap">
        <div
          className="gw-tl-track"
          ref={trackRef}
          onClick={onTrack}
          onMouseMove={e => {
            const r = e.currentTarget.getBoundingClientRect();
            setHover(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)));
          }}
          onMouseLeave={() => setHover(null)}
        >
          <div className="gw-tl-fill" style={{ width: `${t * 100}%` }} />
          {years.map(y => {
            const p = (y - 1985) / 40;
            return (
              <div key={y} className="gw-tl-tick" style={{ left: `${p * 100}%` }}>
                <span>{y}</span>
              </div>
            );
          })}
          {EVENTS.map(e => {
            const row = EVENT_ROWS.get(e.label + e.year) ?? 0;
            return (
              <button
                key={e.label + e.year}
                className={`gw-tl-ev gw-tl-ev-r${row}`}
                style={{ left: `${e.t * 100}%` }}
                onClick={ev => { ev.stopPropagation(); setT(e.t); }}
                title={`${e.label} ${e.year} — ${e.note}`}
              >
                <i />
                <span>{e.label}</span>
              </button>
            );
          })}
          <div className="gw-tl-head-pin" style={{ left: `${t * 100}%` }}>
            <div className={`gw-tl-playhead ${playing ? 'playing' : ''}`} />
          </div>
          {hover !== null && (
            <div className="gw-tl-hover" style={{ left: `${hover * 100}%` }}>
              <span>{Math.floor(1985 + hover * 40)}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
