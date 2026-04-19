"""Generate per-storm validation plots + summary metrics from IBTrACS.

Produces artifacts consumed by the Marimo science-UI notebook:
  artifacts/storm_<name>_<year>.png        2-panel timeline per validation storm
  artifacts/gulf_ri_timeline_40yr.png      40-year Gulf RI event frequency
  artifacts/ri_landfall_scatter.png        RI magnitude vs proximity to coast
  artifacts/validation_summary.json        tabular summary for judges

Run:
    python validation/storm_analysis.py /path/to/ibtracs.NA.list.v04r01.csv
"""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from ibtracs_prep import (
    GULF_LAT_MAX, GULF_LAT_MIN, GULF_LON_MAX, GULF_LON_MIN,
    RI_WIND_DELTA_KT, RI_WINDOW_HOURS,
    find_ri_events, load_gulf_storms, pick_validation_storms,
)

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Brand palette — matches the dashboard's dark-ocean theme.
BG = "#0d1b2a"
PANEL = "#1e293b"
AXIS = "#334155"
TEXT = "#e2e8f0"
ACCENT_RED = "#ef4444"
ACCENT_ORANGE = "#f97316"
ACCENT_BLUE = "#38bdf8"
ACCENT_GREEN = "#22c55e"

plt.rcParams.update({
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "font.size": 10,
    "axes.titleweight": "bold",
})


def _style_ax(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor(AXIS)
    ax.grid(True, alpha=0.15, linestyle=":")


def _rolling_ri_flags(g: pd.DataFrame) -> pd.Series:
    """Boolean series: True where this track point is the start of an RI window."""
    times = g["ISO_TIME"].values
    winds = g["WIND_KT"].values
    flags = np.zeros(len(g), dtype=bool)
    for i in range(len(g)):
        dt_h = (times - times[i]).astype("timedelta64[h]").astype(float)
        later = np.where(dt_h >= RI_WINDOW_HOURS)[0]
        if len(later) == 0:
            break
        j = later[0]
        if winds[j] - winds[i] >= RI_WIND_DELTA_KT:
            flags[i] = True
    return pd.Series(flags, index=g.index)


def plot_storm_timeline(name: str, year: int, g: pd.DataFrame, out_path: str):
    """2-panel: wind-speed + SSHS category over storm lifetime with RI flags."""
    g = g.dropna(subset=["ISO_TIME", "WIND_KT"]).copy()
    if len(g) < 2:
        return None

    g["RI_FLAG"] = _rolling_ri_flags(g)

    # Landfall points have LANDFALL == 0 (on-coast) in IBTrACS conventions.
    landfalls = g[g["LANDFALL"] == 0]
    gulf_mask = (
        (g["LAT"] >= GULF_LAT_MIN) & (g["LAT"] <= GULF_LAT_MAX)
        & (g["LON"] >= GULF_LON_MIN) & (g["LON"] <= GULF_LON_MAX)
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    # ─── Top: wind-speed timeline ────────────────────────────────────────
    ax1.plot(g["ISO_TIME"], g["WIND_KT"], color=ACCENT_ORANGE, lw=2.2,
             label="Max sustained wind (kt)")
    ax1.fill_between(g["ISO_TIME"], 0, g["WIND_KT"],
                     where=gulf_mask, color=ACCENT_ORANGE, alpha=0.18,
                     label="Inside Gulf bbox")

    # RI start markers
    ri_starts = g[g["RI_FLAG"]]
    if len(ri_starts):
        ax1.scatter(ri_starts["ISO_TIME"], ri_starts["WIND_KT"],
                    s=80, marker="^", color=ACCENT_RED, edgecolor="white",
                    lw=1.2, zorder=5,
                    label=f"RI onset (Δ ≥ {int(RI_WIND_DELTA_KT)} kt / 24 h)")

    # Landfall markers
    if len(landfalls):
        for _, r in landfalls.iterrows():
            ax1.axvline(r["ISO_TIME"], color=ACCENT_BLUE, ls="--",
                        lw=1, alpha=0.7)
        ax1.scatter(landfalls["ISO_TIME"], landfalls["WIND_KT"],
                    s=90, marker="X", color=ACCENT_BLUE,
                    edgecolor="white", lw=1.2, zorder=6, label="Landfall")

    # Saffir-Simpson reference bands
    for thresh, label, colour in [
        (64, "Cat 1", "#fde68a"),
        (83, "Cat 2", "#fbbf24"),
        (96, "Cat 3", "#f97316"),
        (113, "Cat 4", "#ef4444"),
        (137, "Cat 5", "#b91c1c"),
    ]:
        ax1.axhline(thresh, color=colour, ls=":", lw=0.7, alpha=0.45)
        ax1.text(g["ISO_TIME"].iloc[0], thresh + 1, label,
                 fontsize=7, color=colour, alpha=0.8)

    ax1.set_ylabel("Wind speed (kt)")
    ax1.set_title(
        f"{name.title()} ({year}) — IBTrACS observed intensity",
        fontsize=13,
    )
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.85,
               facecolor=PANEL, edgecolor=AXIS)
    _style_ax(ax1)
    ax1.set_ylim(0, max(180, g["WIND_KT"].max() + 15))

    # ─── Bottom: pressure + distance-to-land ─────────────────────────────
    ax2.plot(g["ISO_TIME"], g["PRES_MB"], color=ACCENT_BLUE, lw=1.6,
             label="Central pressure (mb)")
    if g["DIST2LAND"].notna().any():
        ax2b = ax2.twinx()
        ax2b.plot(g["ISO_TIME"], g["DIST2LAND"], color=ACCENT_GREEN, lw=1.2,
                  alpha=0.7, label="Distance to land (km)")
        ax2b.set_ylabel("Dist to land (km)", color=ACCENT_GREEN)
        ax2b.tick_params(axis="y", colors=ACCENT_GREEN)
        for sp in ax2b.spines.values():
            sp.set_edgecolor(AXIS)

    ax2.set_ylabel("Pressure (mb)", color=ACCENT_BLUE)
    ax2.tick_params(axis="y", colors=ACCENT_BLUE)
    ax2.set_xlabel("Date (UTC)")
    _style_ax(ax2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # ─── Footer summary ──────────────────────────────────────────────────
    peak_wind = g["WIND_KT"].max()
    min_pres = g["PRES_MB"].min()
    n_ri = int(g["RI_FLAG"].sum())
    fig.text(
        0.01, 0.01,
        f"Peak wind {peak_wind:.0f} kt  ·  Min pressure {min_pres:.0f} mb"
        f"  ·  {n_ri} RI onset window(s)",
        fontsize=9, color=TEXT, alpha=0.85,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "name": name,
        "year": year,
        "peak_wind_kt": float(peak_wind),
        "min_pres_mb": float(min_pres) if not pd.isna(min_pres) else None,
        "n_ri_windows": n_ri,
        "first_obs": str(g["ISO_TIME"].iloc[0]),
        "last_obs": str(g["ISO_TIME"].iloc[-1]),
        "n_landfalls": int(len(landfalls)),
    }


def plot_40yr_ri_timeline(ri_events: pd.DataFrame, out_path: str):
    """Bar chart of Gulf RI events per year — the 'climate' context plot."""
    gulf = ri_events[ri_events["in_gulf"]].copy()
    if len(gulf) == 0:
        return
    yearly = gulf.groupby("SEASON").size().reindex(range(1982, 2025), fill_value=0)

    fig, ax = plt.subplots(figsize=(13, 4.2))
    colors = [ACCENT_RED if y >= yearly.mean() + yearly.std() else ACCENT_ORANGE
              for y in yearly.values]
    ax.bar(yearly.index, yearly.values, color=colors, edgecolor=BG, lw=0.5)

    # Trend line
    if yearly.sum() > 0:
        z = np.polyfit(yearly.index, yearly.values, 1)
        trend = np.poly1d(z)(yearly.index)
        ax.plot(yearly.index, trend, color=ACCENT_BLUE, lw=2, ls="--",
                label=f"Linear trend: {z[0]:+.2f} events/year/decade")
        ax.legend(loc="upper left", fontsize=9, facecolor=PANEL, edgecolor=AXIS)

    # Annotate the peak year
    peak_year = yearly.idxmax()
    ax.annotate(
        f"{peak_year}: {int(yearly.max())} RI episodes",
        xy=(peak_year, yearly.max()),
        xytext=(peak_year - 8, yearly.max() + 0.5),
        fontsize=9, color=TEXT,
        arrowprops=dict(arrowstyle="->", color=AXIS),
    )

    ax.set_xlabel("Season")
    ax.set_ylabel("Gulf RI episodes")
    ax.set_title(
        "40 Years of Gulf of Mexico Rapid Intensification — IBTrACS",
        fontsize=12,
    )
    _style_ax(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_ri_magnitude_scatter(ri_events: pd.DataFrame, out_path: str):
    """RI delta vs onset location — shows the LC-eddy corridor."""
    gulf = ri_events[ri_events["in_gulf"]].copy()
    if len(gulf) == 0:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    sc = ax.scatter(
        gulf["lon"], gulf["lat"],
        c=gulf["delta_kt"], s=gulf["delta_kt"] * 2,
        cmap="YlOrRd", vmin=30, vmax=80,
        edgecolor="white", lw=0.5, alpha=0.85,
    )

    # Label the strongest five RI events
    top = gulf.nlargest(5, "delta_kt")
    for _, r in top.iterrows():
        ax.annotate(
            f"{r['NAME'].title()} {int(r['SEASON'])}  (+{int(r['delta_kt'])} kt)",
            xy=(r["lon"], r["lat"]), xytext=(5, 5),
            textcoords="offset points", fontsize=8.5, color=TEXT,
        )

    # Loop Current eddy-shedding corridor (approximate)
    ax.add_patch(plt.Rectangle(
        (-87, 23), 4, 4, fill=False,
        edgecolor=ACCENT_BLUE, lw=1.8, ls="--", alpha=0.7,
    ))
    ax.text(-87, 27.3, "Loop Current eddy-shedding zone",
            color=ACCENT_BLUE, fontsize=9, alpha=0.9)

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Wind speed Δ over 24 h (kt)", color=TEXT)
    cb.ax.yaxis.set_tick_params(color=TEXT)

    ax.set_xlim(GULF_LON_MIN, GULF_LON_MAX)
    ax.set_ylim(GULF_LAT_MIN, GULF_LAT_MAX)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Where Gulf storms rapidly intensify — {len(gulf)} events, 1982–2024",
        fontsize=12,
    )
    _style_ax(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(csv_path: str):
    print(f"Loading {csv_path} …")
    storms = load_gulf_storms(csv_path)
    print(f"  {len(storms)} Gulf-touching storms since 1982")

    ri = find_ri_events(storms)
    gulf_ri = ri[ri["in_gulf"]].copy() if len(ri) else ri
    print(f"  {len(ri)} RI episodes total · {len(gulf_ri)} inside Gulf")

    print("\nGenerating 40-year context plots …")
    plot_40yr_ri_timeline(ri, os.path.join(ARTIFACT_DIR, "gulf_ri_timeline_40yr.png"))
    plot_ri_magnitude_scatter(ri, os.path.join(ARTIFACT_DIR, "ri_landfall_scatter.png"))
    print("  ✓ gulf_ri_timeline_40yr.png")
    print("  ✓ ri_landfall_scatter.png")

    print("\nGenerating per-storm validation plots …")
    picks = pick_validation_storms(storms)
    per_storm: list[dict] = []
    for key, g in picks.items():
        name, year = key.split("_")
        out = os.path.join(ARTIFACT_DIR, f"storm_{name.lower()}_{year}.png")
        meta = plot_storm_timeline(name, int(year), g, out)
        if meta:
            per_storm.append(meta)
            print(f"  ✓ {os.path.basename(out)}  "
                  f"(peak {meta['peak_wind_kt']:.0f} kt, "
                  f"{meta['n_ri_windows']} RI onset(s))")

    # Summary JSON for the Marimo notebook and the API-layer metadata badge.
    summary = {
        "source": "NOAA IBTrACS North Atlantic v04r01",
        "bbox": {
            "lat": [GULF_LAT_MIN, GULF_LAT_MAX],
            "lon": [GULF_LON_MIN, GULF_LON_MAX],
        },
        "ri_definition": {
            "wind_delta_kt": RI_WIND_DELTA_KT,
            "window_hours": RI_WINDOW_HOURS,
            "reference": "NHC standard (≥ 30 kt / 24 h)",
        },
        "counts": {
            "gulf_storms_since_1982": len(storms),
            "ri_episodes_total": int(len(ri)),
            "ri_episodes_gulf": int(len(gulf_ri)),
            "years_covered": [1982, 2024],
        },
        "top_gulf_ri_events": (
            gulf_ri.nlargest(10, "delta_kt")[
                ["NAME", "SEASON", "start_time", "wind_start", "wind_end", "delta_kt"]
            ]
            .assign(start_time=lambda d: d["start_time"].astype(str))
            .to_dict(orient="records")
            if len(gulf_ri) else []
        ),
        "validation_storms": per_storm,
    }

    summary_path = os.path.join(ARTIFACT_DIR, "validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  ✓ {os.path.basename(summary_path)}")
    print(f"\nAll artifacts written to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "/Users/rishil/Desktop/ibtracs.NA.list.v04r01.csv"
    run(csv)
