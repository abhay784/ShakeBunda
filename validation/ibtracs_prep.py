"""IBTrACS data preparation for Gulf Watch validation.

Loads the NOAA IBTrACS North Atlantic CSV, filters to the Gulf of Mexico,
and flags rapid-intensification (RI) episodes using the NHC definition
(≥ 30 kt wind increase over 24 h).

Usage:
    from ibtracs_prep import load_gulf_storms, find_ri_events
    storms = load_gulf_storms("/path/to/ibtracs.NA.list.v04r01.csv")
    ri = find_ri_events(storms)
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

# Gulf of Mexico bounding box (slightly larger than the model grid so storms
# that intensify offshore before entering the Gulf proper are still captured).
GULF_LAT_MIN, GULF_LAT_MAX = 18.0, 32.0
GULF_LON_MIN, GULF_LON_MAX = -98.0, -78.0

# NHC threshold: 30 kt wind-speed increase in any rolling 24-hour window.
RI_WIND_DELTA_KT = 30.0
RI_WINDOW_HOURS = 24

VALIDATION_STORMS = {
    "Katrina": 2005,
    "Rita":    2005,
    "Harvey":  2017,
    "Ida":     2021,
    "Michael": 2018,
}


def load_ibtracs(csv_path: str) -> pd.DataFrame:
    """Load IBTrACS North Atlantic CSV. Skips the units-header row."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"IBTrACS CSV not found: {csv_path}")

    # Row 0 is the data header, row 1 is the units row — skip it.
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False,
                     na_values=[" ", "", "NA"])

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    for c in ("LAT", "LON", "USA_WIND", "WMO_WIND", "USA_PRES", "WMO_PRES",
             "LANDFALL", "DIST2LAND"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Prefer NHC (USA) wind speed for NA basin; fall back to WMO.
    df["WIND_KT"] = df["USA_WIND"].fillna(df["WMO_WIND"])
    df["PRES_MB"] = df["USA_PRES"].fillna(df["WMO_PRES"])
    return df


def filter_to_gulf(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only track points whose centre falls inside the Gulf bbox."""
    mask = (
        (df["LAT"] >= GULF_LAT_MIN) & (df["LAT"] <= GULF_LAT_MAX) &
        (df["LON"] >= GULF_LON_MIN) & (df["LON"] <= GULF_LON_MAX)
    )
    return df.loc[mask].copy()


def load_gulf_storms(csv_path: str, min_year: int = 1982) -> dict[str, pd.DataFrame]:
    """Return {storm_id: per-storm time-sorted dataframe} for Gulf storms."""
    df = load_ibtracs(csv_path)
    df = df[df["SEASON"] >= min_year]

    gulf = filter_to_gulf(df)
    # Keep any storm that entered the Gulf, not only points inside.
    gulf_ids = gulf["SID"].unique()
    storms = df[df["SID"].isin(gulf_ids)].sort_values(["SID", "ISO_TIME"])

    return {sid: g.reset_index(drop=True) for sid, g in storms.groupby("SID")}


def find_ri_events(storms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flag 24-hour RI episodes (≥ 30 kt wind-speed increase).

    Returns a dataframe with one row per RI episode:
        SID, NAME, SEASON, start_time, end_time, wind_start, wind_end, delta_kt,
        lat, lon (at midpoint), in_gulf (bool)
    """
    rows = []
    for sid, g in storms.items():
        g = g.dropna(subset=["WIND_KT", "ISO_TIME"]).copy()
        if len(g) < 2:
            continue

        name = g["NAME"].dropna().iloc[0] if g["NAME"].notna().any() else "UNNAMED"
        season = int(g["SEASON"].iloc[0])

        times = g["ISO_TIME"].values
        winds = g["WIND_KT"].values
        lats = g["LAT"].values
        lons = g["LON"].values

        for i in range(len(g)):
            t0 = times[i]
            # Find the first point >= 24h later
            dt_ns = (times - t0).astype("timedelta64[h]").astype(float)
            later = np.where(dt_ns >= RI_WINDOW_HOURS)[0]
            if len(later) == 0:
                break
            j = later[0]
            delta = winds[j] - winds[i]
            if delta >= RI_WIND_DELTA_KT:
                mid_lat = (lats[i] + lats[j]) / 2
                mid_lon = (lons[i] + lons[j]) / 2
                in_gulf = (
                    GULF_LAT_MIN <= mid_lat <= GULF_LAT_MAX
                    and GULF_LON_MIN <= mid_lon <= GULF_LON_MAX
                )
                rows.append({
                    "SID": sid,
                    "NAME": name,
                    "SEASON": season,
                    "start_time": pd.Timestamp(t0),
                    "end_time": pd.Timestamp(times[j]),
                    "wind_start": float(winds[i]),
                    "wind_end": float(winds[j]),
                    "delta_kt": float(delta),
                    "lat": float(mid_lat),
                    "lon": float(mid_lon),
                    "in_gulf": bool(in_gulf),
                })

    ri = pd.DataFrame(rows)
    if len(ri) == 0:
        return ri

    # Deduplicate: a storm intensifying steadily triggers RI at consecutive
    # time steps. Keep the strongest episode per 48h window per storm.
    ri = ri.sort_values(["SID", "delta_kt"], ascending=[True, False])
    dedup = []
    for sid, g in ri.groupby("SID"):
        used_times: list[pd.Timestamp] = []
        for _, row in g.iterrows():
            if any(abs((row["start_time"] - t).total_seconds()) < 48 * 3600
                   for t in used_times):
                continue
            dedup.append(row)
            used_times.append(row["start_time"])
    return pd.DataFrame(dedup).sort_values("start_time").reset_index(drop=True)


def pick_validation_storms(
    storms: dict[str, pd.DataFrame],
    names: Iterable[tuple[str, int]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return a subset of storms matching (name, season) pairs."""
    if names is None:
        names = list(VALIDATION_STORMS.items())
    picks: dict[str, pd.DataFrame] = {}
    for name, year in names:
        for sid, g in storms.items():
            storm_name = g["NAME"].dropna().iloc[0] if g["NAME"].notna().any() else ""
            storm_year = int(g["SEASON"].iloc[0])
            if str(storm_name).upper() == name.upper() and storm_year == year:
                picks[f"{name}_{year}"] = g
                break
    return picks


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "/Users/rishil/Desktop/ibtracs.NA.list.v04r01.csv"
    print(f"Loading {csv} …")
    storms = load_gulf_storms(csv)
    print(f"  Gulf-touching storms since 1982: {len(storms)}")

    ri = find_ri_events(storms)
    gulf_ri = ri[ri["in_gulf"]] if len(ri) else ri
    print(f"  RI episodes total:          {len(ri)}")
    print(f"  RI episodes inside Gulf:    {len(gulf_ri)}")

    if len(gulf_ri) > 0:
        print("\nTop 10 Gulf RI events by wind-speed delta:")
        print(gulf_ri.nlargest(10, "delta_kt")[
            ["NAME", "SEASON", "start_time", "wind_start", "wind_end", "delta_kt"]
        ].to_string(index=False))

    picks = pick_validation_storms(storms)
    print(f"\nValidation storms found: {list(picks.keys())}")
