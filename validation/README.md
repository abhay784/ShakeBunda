# Gulf Watch — IBTrACS Validation Pipeline

External validation of the CNN + LSTM LCE-separation model against NOAA's
International Best Track Archive for Climate Stewardship (IBTrACS) — the
authoritative hurricane-intensity record used in hurricane-research publications.

## Why this exists

The model in `databricks/gulf_watch_notebook.py` is trained with **self-supervised**
labels derived from SSH thresholds. Those labels are reasonable (Mainelli et al. 2008)
but they are not independent observations. Scripps judges will (rightly) ask:

> *Does the model's prediction timing actually match real hurricane rapid-intensification
> events in the Gulf?*

This pipeline answers that question by cross-referencing five well-known validation
storms (Katrina, Rita, Harvey, Ida, Michael) against the official NHC best-track record.

## Files

| Path | Purpose |
|------|---------|
| `ibtracs_prep.py` | Loader, Gulf-bbox filter, 24-hour RI-window detector |
| `storm_analysis.py` | Generates per-storm plots + 40-year climatology |
| `artifacts/validation_summary.json` | Tabular summary consumed by Marimo |
| `artifacts/storm_*.png` | Per-storm intensity timelines with RI / landfall overlays |
| `artifacts/gulf_ri_timeline_40yr.png` | 40-year Gulf RI event frequency bar chart |
| `artifacts/ri_landfall_scatter.png` | RI magnitude vs. location (shows LC corridor) |

## Setup

```bash
# 1. Download the NA basin CSV (~330 MB → 57 MB after gzip):
curl -O https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv

# 2. Install deps (pandas + numpy + matplotlib — no scipy, no cartopy):
pip install pandas numpy matplotlib
```

## Run

```bash
# Defaults to /Users/rishil/Desktop/ibtracs.NA.list.v04r01.csv
python validation/storm_analysis.py

# Or point at a custom location:
python validation/storm_analysis.py /path/to/ibtracs.NA.list.v04r01.csv
```

Expected output:

```
Loading /Users/rishil/Desktop/ibtracs.NA.list.v04r01.csv …
  275 Gulf-touching storms since 1982
  108 RI episodes total · 63 inside Gulf

Generating 40-year context plots …
  ✓ gulf_ri_timeline_40yr.png
  ✓ ri_landfall_scatter.png

Generating per-storm validation plots …
  ✓ storm_katrina_2005.png  (peak 150 kt, 8 RI onset(s))
  ✓ storm_rita_2005.png     (peak 155 kt, 13 RI onset(s))
  ✓ storm_harvey_2017.png   (peak 115 kt, 14 RI onset(s))
  ✓ storm_ida_2021.png      (peak 130 kt, 11 RI onset(s))
  ✓ storm_michael_2018.png  (peak 140 kt, 14 RI onset(s))
```

## Rapid-intensification definition

NHC standard: **wind-speed increase ≥ 30 kt in any 24-hour window.** Implemented in
`ibtracs_prep.find_ri_events()` — for each track point, looks forward to the first
observation ≥ 24 h later and compares wind-speed delta. Consecutive overlapping
windows are deduplicated (48-hour cooldown, keeping the strongest episode per storm).

Wind speed preference:
1. `USA_WIND` (NHC best track) — authoritative for North Atlantic
2. `WMO_WIND` fallback where NHC is missing

## Gulf of Mexico bounding box

Intentionally wider than the ECCO model grid so storms that rapidly intensify just
*before* entering the Gulf proper are still counted:

| | min | max |
|---|---|---|
| Latitude | 18 °N | 32 °N |
| Longitude | −98 °E | −78 °E |

## Hooking into the Marimo notebook

`marimo_notebook.py` reads `artifacts/validation_summary.json` and the PNG files on
import. No additional wiring needed — just regenerate the artifacts and reload.

## Hooking into the Databricks notebook

`databricks/gulf_watch_notebook.py` **CELL 11 — IBTrACS EXTERNAL VALIDATION** performs
the same analysis but with access to the live trained model, so it can emit real
hit/miss counts for each RI onset window and a precision/recall score. Upload the CSV
to `/dbfs/FileStore/gulf_watch/ibtracs.NA.list.v04r01.csv` first.

## Interpreting the per-storm plots

Each plot has two panels:

- **Top** — observed max sustained wind over the storm's lifetime.
  - **Orange fill** = inside Gulf bbox
  - **Red triangles** = start of a 24-hour RI window (≥ 30 kt increase)
  - **Blue X markers** = landfall
  - Dotted horizontal lines = Saffir-Simpson category thresholds

- **Bottom** — central pressure (blue) and distance to land (green).

The per-storm timeline is *independent of the CNN + LSTM model output*. When the model's
`lce_separation_prob_7d` series is also available (Databricks Cell 11), both curves are
overlaid in the output PNG so you can visually check that probability rises ahead of the
IBTrACS-observed RI onsets.

## References

- Knapp, K. R. et al. (2010). *The International Best Track Archive for Climate Stewardship
  (IBTrACS).* Bull. Amer. Meteor. Soc., 91, 363–376.
- Kaplan & DeMaria (2003). *Large-scale characteristics of rapidly intensifying tropical
  cyclones in the North Atlantic basin.* Wea. Forecasting.
- Mainelli, M. et al. (2008). *Application of oceanic heat content to operational forecasting
  of hurricane intensity.* Wea. Forecasting, 23, 3–16.
- Shay, L. K. et al. (2000). *Effects of a warm oceanic feature on Hurricane Opal.*
  Mon. Wea. Rev., 128, 1366–1383.
