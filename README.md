# Precipitation and TTC weather-related bus delays

Interactive Streamlit dashboard: daily precipitation across Toronto (inverse distance weighted from the
City of Toronto rain gauges) with the TTC's weather-related bus delays.

* Pick a date, or jump between delay days, to map that day's precipitation surface and delays
* Click a day on the timeline or a row in the delay-day table to map it
* Compare delays per day across precipitation groups
* Advanced settings (sidebar): winter gauge rule, IDW power, colour scale, hot-spot layer

## Pages

* **Precipitation & weather delays** (`precipitation_delays.py`): the interactive map and charts above.
* **Bus delay EDA (pre-2025)** (`bus_delay_eda.py`): incident reasons for all TTC bus delays before 2025
  and a hot-spot map of the four focus reasons (General Delay, Operations, Diversion, Other).
  It reads two small files made by `prepare_bus_eda_data.py`:

      python prepare_bus_eda_data.py pre_2025_ttc_bus_delays_cleaned.csv pre_2025_ttc_bus_delays_with_coords.csv

  which writes `data/bus_incident_reasons_by_year.csv` and `data/bus_focus_delays_with_coords.csv`.
  Upload those two files to the `data` folder; the large original CSVs are not needed online.

## Run locally

    pip install -r requirements.txt
    streamlit run streamlit_app.py

## Deploy on Streamlit Community Cloud

1. Put this folder's contents in a public GitHub repository (keep the `data` and `.streamlit` folders).
2. At https://share.streamlit.io choose **Create app → Deploy a public app from GitHub**,
   pick the repository and branch, and set the main file to `streamlit_app.py`.

## Data (`data/`)

| File | Contents |
|---|---|
| `daily_precipitation.csv` | Daily precipitation (mm) per gauge, Jun 2015 – Oct 2026, summed from Toronto Open Data 5-minute readings |
| `rain_gauges.csv` | Gauge locations and 3-/4-season type |
| `bus_delays.csv` | TTC weather-related bus delays with coordinates |
| `city_grid.csv` | ~500 m grid cells covering the city, with neighbourhood |
| `neighbourhoods.json` | Toronto neighbourhood outlines (simplified) |
