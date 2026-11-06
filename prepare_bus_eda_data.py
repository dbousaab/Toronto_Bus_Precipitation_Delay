"""
Prepare the small data files used by the "Bus delay EDA (pre-2025)" page.

Run once, in the folder that holds the two large CSVs from pre_2025_ttc_bus_EDA.ipynb:
    python prepare_bus_eda_data.py  path/to/pre_2025_ttc_bus_delays_cleaned.csv  path/to/pre_2025_ttc_bus_delays_with_coords.csv
(with no arguments it looks for both files in the current folder)

It writes into ./data next to this script:
    bus_incident_reasons_by_year.csv   delays per incident reason and year (all reasons)
    bus_focus_delays_with_coords.csv   delays for the four focus reasons with date, time and coordinates
"""
import sys
from pathlib import Path

import pandas as pd

FOCUS_REASONS = ["Other", "General Delay", "Diversion", "Operations"]
HERE = Path(__file__).resolve().parent
OUT = HERE / "data"

cleaned_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pre_2025_ttc_bus_delays_cleaned.csv")
coords_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("pre_2025_ttc_bus_delays_with_coords.csv")
OUT.mkdir(exist_ok=True)

# 1. Incident reasons (all delays), counted per year
cleaned = pd.read_csv(cleaned_path, usecols=lambda c: c in {"Date", "Incident", "Min Delay"}, low_memory=False)
cleaned["year"] = pd.to_datetime(cleaned["Date"], errors="coerce").dt.year
cleaned["Incident"] = cleaned["Incident"].fillna("Unknown").astype(str).str.strip()
reasons = (cleaned.dropna(subset=["year"])
           .groupby(["year", "Incident"])
           .agg(delays=("Incident", "size"),
                delay_minutes=("Min Delay", lambda s: pd.to_numeric(s, errors="coerce").sum()))
           .reset_index().rename(columns={"Incident": "incident"}))
reasons["year"] = reasons["year"].astype(int)
reasons["delay_minutes"] = reasons["delay_minutes"].round().astype(int)
reasons.to_csv(OUT / "bus_incident_reasons_by_year.csv", index=False)
print(f"Wrote bus_incident_reasons_by_year.csv: {len(cleaned):,} delays, {reasons['incident'].nunique()} reasons")

# 2. Focus-reason delays with coordinates (for the hot-spot map)
keep = {"Date", "Datetime", "Incident", "Route", "Location", "Station", "Min Delay", "Latitude", "Longitude"}
coords = pd.read_csv(coords_path, usecols=lambda c: c in keep, low_memory=False)
coords = coords[coords["Incident"].isin(FOCUS_REASONS)].copy()
coords["Latitude"] = pd.to_numeric(coords["Latitude"], errors="coerce")
coords["Longitude"] = pd.to_numeric(coords["Longitude"], errors="coerce")
coords = coords.dropna(subset=["Latitude", "Longitude"])
coords = coords[coords["Latitude"].between(43.4, 44.2) & coords["Longitude"].between(-80.0, -78.8)]
when = pd.to_datetime(coords["Datetime"] if "Datetime" in coords else coords["Date"], errors="coerce")
place = coords["Location"] if "Location" in coords else coords.get("Station", "")
out = pd.DataFrame({
    "date": when.dt.strftime("%Y-%m-%d"),
    "time": when.dt.strftime("%H:%M"),
    "incident": coords["Incident"],
    "route": coords["Route"].astype(str) if "Route" in coords else "",
    "location": place,
    "delay_min": pd.to_numeric(coords.get("Min Delay", 0), errors="coerce").fillna(0).astype(int),
    "latitude": coords["Latitude"].round(5),
    "longitude": coords["Longitude"].round(5),
}).dropna(subset=["date"])
out.to_csv(OUT / "bus_focus_delays_with_coords.csv", index=False)
size_mb = (OUT / "bus_focus_delays_with_coords.csv").stat().st_size / 1e6
print(f"Wrote bus_focus_delays_with_coords.csv: {len(out):,} delays with coordinates ({size_mb:.1f} MB)")
if size_mb > 24:
    print("Note: GitHub's web upload takes files up to 25 MB; this one is larger.")
