"""One-time preprocessing script — runs on your laptop, not on Databricks.

Reads run2_clim_v2_ssh.nc from the given path, crops to the Gulf of Mexico,
computes the SSH anomaly (ssh - time-mean), and writes artifacts/ssh_anomaly.npz
in the format expected by databricks/server.py.

The model weights and metadata.json (which now contains std_ch) must already
be in the artifacts/ directory — download them from Databricks Workspace first.

Usage:
    python -m databricks.preprocess_local ~/Downloads/run2_clim_v2_ssh.nc
    # or with a custom output dir:
    python -m databricks.preprocess_local ~/Downloads/run2_clim_v2_ssh.nc --out ./artifacts

The script validates that the grid shape (H, W) matches the h/w stored in
metadata.json. If they don't match the model weights are incompatible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Gulf of Mexico bounding box (degrees)
_LAT_MIN, _LAT_MAX = 22.0, 31.0
_LON_MIN_NEG, _LON_MAX_NEG = -97.0, -80.0  # ±180° convention
_LON_MIN_POS, _LON_MAX_POS = 263.0, 280.0  # 0–360° convention


def _detect(ds, candidates: list[str], kind: str) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    print(f"  Could not find {kind} coordinate. Available: {list(ds.coords)}")
    raise KeyError(f"No {kind} coordinate found in {candidates}")


def preprocess(nc_path: Path, out_dir: Path) -> None:
    try:
        import xarray as xr
    except ImportError:
        sys.exit("xarray not installed — run: pip install xarray netCDF4")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ssh_anomaly.npz"

    print(f"Opening {nc_path} …")
    # decode_times=False: the ECCO file stores time as "days since Jan-1-0000"
    # (year zero), which pandas and cftime both fail to parse. We decode manually
    # below after detecting the time dimension name.
    # No chunks= here — dask not required. xarray opens the file lazily via
    # netCDF4; only the Gulf crop gets pulled into RAM when .load() is called.
    ds = xr.open_dataset(nc_path, decode_times=False)

    # --- variable detection --------------------------------------------------
    ssh_var = None
    for v in ["ssh", "eta", "zos", "sea_surface_height", "SSH", "ETA", "adt"]:
        if v in ds.data_vars:
            ssh_var = v
            break
    if ssh_var is None:
        print("  Available data_vars:", list(ds.data_vars))
        sys.exit("No SSH variable found. Add the correct name to the loop above.")
    print(f"  SSH variable: {ssh_var!r}")

    # --- coordinate detection ------------------------------------------------
    lat_dim = _detect(ds, ["lat", "latitude", "y", "YC"], "latitude")
    lon_dim = _detect(ds, ["lon", "longitude", "x", "XC"], "longitude")
    time_dim = _detect(ds, ["time", "TIME", "t"], "time")
    print(f"  Coordinates: lat={lat_dim!r}, lon={lon_dim!r}, time={time_dim!r}")

    # --- manual time decoding ------------------------------------------------
    # Units are "days since Jan-1-0000". Year zero is unparseable by pandas and
    # problematic in cftime. Use pure numpy arithmetic instead:
    #   days from 0000-01-01 → days from 1970-01-01 = subtract 719528
    #   (year 0 = 366 days proleptic leap + 719162 days years 1–1969 to 1970)
    # Strip the numpy mask first — time var has _FillValue=-9999 in attrs.
    time_raw = np.ma.getdata(ds[time_dim].values).astype(np.float64)
    # Exclude any genuine fill values before converting
    valid = time_raw > 0
    time_raw = time_raw[valid]
    _DAYS_0000_TO_1970 = np.int64(719528)
    days_since_epoch = (time_raw - _DAYS_0000_TO_1970).astype(np.int64)
    dates_np = (np.datetime64("1970-01-01", "D") +
                days_since_epoch.astype("timedelta64[D]"))
    times_decoded = dates_np.astype("datetime64[D]").astype(str)
    print(f"  Time range: {times_decoded[0]} → {times_decoded[-1]} ({len(times_decoded)} frames)")

    # --- longitude convention ------------------------------------------------
    if float(ds[lon_dim].max()) > 180:
        lon_lo, lon_hi = _LON_MIN_POS, _LON_MAX_POS
        print("  Longitude convention: 0–360°  → cropping to 263–280°E")
    else:
        lon_lo, lon_hi = _LON_MIN_NEG, _LON_MAX_NEG
        print("  Longitude convention: ±180°   → cropping to -97–-80°")

    # --- Gulf crop (load only this slice into RAM) ----------------------------
    print("  Cropping to Gulf of Mexico …")
    gulf = ds[ssh_var].sel(
        {lat_dim: slice(_LAT_MIN, _LAT_MAX), lon_dim: slice(lon_lo, lon_hi)}
    )
    print(f"  Gulf shape before load: {dict(zip(gulf.dims, gulf.shape))}")
    ssh = gulf.load()  # only ~few hundred MB for the crop
    print(f"  Loaded. shape={ssh.shape}, dtype={ssh.dtype}")

    # --- SSH anomaly ----------------------------------------------------------
    # Subtract time-mean — must match training notebook exactly.
    print("  Computing anomaly (ssh - time mean) …")
    mean = ssh.mean(dim=time_dim)
    anomaly = ssh - mean

    # Fill NaN with 0 (coastal/land cells that remain masked).
    raw = np.nan_to_num(anomaly.values.astype(np.float32), nan=0.0)
    times = times_decoded  # already decoded above as 'YYYY-MM-DD' strings
    lats = ssh[lat_dim].values.astype(np.float32)
    lons = ssh[lon_dim].values.astype(np.float32)

    T, H, W = raw.shape
    print(f"  Final cube: T={T}, H={H}, W={W}")

    # --- validate against metadata.json if present ---------------------------
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        mH, mW = meta.get("h"), meta.get("w")
        if (mH, mW) != (H, W):
            print(
                f"\n  WARNING: grid shape mismatch!\n"
                f"    metadata.json says h={mH}, w={mW}\n"
                f"    local crop produced H={H}, W={W}\n"
                f"  The encoder FC layer was trained on ({mH}×{mW}).\n"
                f"  Check that you're using the same crop bounds and NetCDF file.\n"
                f"  Proceeding — but predictions will be wrong until this is fixed."
            )
        else:
            print(f"  Grid shape ({H}×{W}) matches metadata.json ✓")
    else:
        print("  metadata.json not found in out_dir — skipping shape validation.")

    # --- write ---------------------------------------------------------------
    print(f"  Writing {out_path} …")
    np.savez_compressed(
        out_path,
        raw=raw,
        times=times,
        lats=lats,
        lons=lons,
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Done. {out_path}  ({size_mb:.1f} MB)")
    print()
    print("Next step: verify artifacts/ contains lce_predictor.pt, metadata.json,")
    print("and ssh_anomaly.npz, then run:")
    print("  GULF_WATCH_ARTIFACTS_DIR=./artifacts uvicorn databricks.server:app --port 8000")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Gulf SSH NetCDF for local serving.")
    parser.add_argument("nc_path", type=Path, help="Path to run2_clim_v2_ssh.nc")
    parser.add_argument(
        "--out", type=Path, default=Path("./artifacts"),
        help="Output directory (default: ./artifacts)",
    )
    args = parser.parse_args()

    if not args.nc_path.exists():
        sys.exit(f"NetCDF not found: {args.nc_path}")

    preprocess(args.nc_path, args.out)


if __name__ == "__main__":
    main()
