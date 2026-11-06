"""
Precipitation and TTC weather-related bus delays - Streamlit dashboard

Local run:      pip install -r requirements.txt
                streamlit run streamlit_app.py
Deploy:         push this folder to a GitHub repository, then create the app on
                https://share.streamlit.io with main file "streamlit_app.py".

Data (in ./data, prepared from the project's precipitation notebook):
    daily_precipitation.csv    daily precipitation total (mm) per City of Toronto rain gauge, Jun 2015 - Oct 2026
    rain_gauges.csv       gauge locations and 3-/4-season type
    bus_delays.csv        TTC weather-related bus delays with coordinates
    city_grid.csv         ~500 m grid cells covering the city, tagged with neighbourhood
    neighbourhoods.json   Toronto neighbourhood outlines (simplified)
"""
import json
from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"  # same folder as streamlit_app.py
WINTER_MONTHS = [11, 12, 1, 2, 3]
LAT0 = 43.7
M_LAT = 110574.0                                   # metres per degree of latitude
M_LON = 111320.0 * np.cos(np.radians(LAT0))        # metres per degree of longitude at Toronto
PRECIP_BLUE = "#1d91c0"
DELAY_ORANGE = "#e8590c"
PRECIP_RAMP = ["#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"]
PRECIP_GROUP_BINS = [-np.inf, 0.2, 2, 5, 10, 25, np.inf]
PRECIP_GROUP_LABELS = ["Dry (under 0.2 mm)", "0.2–2 mm", "2–5 mm", "5–10 mm", "10–25 mm", "25 mm or more"]



# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    precip = pd.read_csv(DATA / "daily_precipitation.csv", parse_dates=["date"], index_col="date").sort_index()
    gauges = pd.read_csv(DATA / "rain_gauges.csv")
    precip = precip.reindex(columns=gauges["name"])
    delays = pd.read_csv(DATA / "bus_delays.csv", dtype={"route": str, "direction": str})
    delays["date"] = pd.to_datetime(delays["date"])
    delays[["route", "direction", "location"]] = delays[["route", "direction", "location"]].fillna("")
    grid = pd.read_csv(DATA / "city_grid.csv")
    neighbourhoods = json.loads((DATA / "neighbourhoods.json").read_text(encoding="utf-8"))
    return precip, gauges, delays, grid, neighbourhoods


def apply_season_rule(precip, gauges, winter_rule):
    """Nov–Mar: blank out the 3-season gauges so only the 4-season gauges are counted."""
    counted = precip.copy()
    if winter_rule:
        winter = counted.index.month.isin(WINTER_MONTHS)
        counted.loc[winter, gauges.loc[gauges["season"] == "3-season", "name"]] = np.nan
    return counted


def build_daily_summary(counted, delays):
    per_day = delays.groupby("date").agg(delays=("delay_id", "size"), delay_minutes=("delay_min", "sum"),
                                         routes=("route", "nunique"))
    has_data = counted.notna().any(axis=1)
    summary = pd.DataFrame({
        "gauges_counted": counted.notna().sum(axis=1),
        "mean_precip_mm": counted.mean(axis=1).round(1),
        "max_precip_mm": counted.max(axis=1).round(1),
        "wettest_gauge": counted.fillna(-1).idxmax(axis=1).where(has_data),
    }).join(per_day, how="left").fillna({"delays": 0, "delay_minutes": 0, "routes": 0})
    summary[["delays", "delay_minutes", "routes"]] = summary[["delays", "delay_minutes", "routes"]].astype(int)
    summary.index.name = "date"
    return summary


def build_precip_groups(summary, delays):
    in_record = summary.loc[delays["date"].min():delays["date"].max()].dropna(subset=["mean_precip_mm"])
    groups = (in_record.assign(precip_group=pd.cut(in_record["mean_precip_mm"], PRECIP_GROUP_BINS,
                                                 labels=PRECIP_GROUP_LABELS, right=False))
              .groupby("precip_group", observed=False)
              .agg(days=("delays", "size"), delays=("delays", "sum")))
    groups["delays_per_day"] = (groups["delays"] / groups["days"]).round(2)
    groups = groups.reset_index()
    groups["precip_group"] = groups["precip_group"].astype(str)
    return groups


# --------------------------------------------------------------------------------------
# Distances and inverse distance weighting
# --------------------------------------------------------------------------------------
def to_xy(lat, lon):
    return np.column_stack([np.asarray(lon) * M_LON, np.asarray(lat) * M_LAT])


def distance_matrix(from_xy, to_xy_):
    return np.sqrt(((from_xy[:, None, :] - to_xy_[None, :, :]) ** 2).sum(axis=2))


def idw_estimate(distances, values, power=2.0):
    """Inverse distance weighting; gauges with no reading (NaN) are left out."""
    values = np.asarray(values, dtype=float)
    observed = ~np.isnan(values)
    if not observed.any():
        return np.full(distances.shape[0], np.nan)
    d, v = distances[:, observed], values[observed]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / np.power(d, power)
        estimate = (w @ v) / w.sum(axis=1)
    on_gauge = d < 1.0
    hit = on_gauge.any(axis=1)
    estimate[hit] = v[on_gauge.argmax(axis=1)[hit]]
    return estimate


def point_in_rings(lon, lat, rings):
    """Even-odd ray casting for many points against one neighbourhood's rings."""
    inside = np.zeros(len(lon), dtype=bool)
    for ring in rings:
        r = np.asarray(ring)
        xs, ys = r[:, 0], r[:, 1]
        j = len(r) - 1
        for i in range(len(r)):
            with np.errstate(divide="ignore", invalid="ignore"):
                cross = ((ys[i] > lat) != (ys[j] > lat)) & (lon < (xs[j] - xs[i]) * (lat - ys[i]) / (ys[j] - ys[i]) + xs[i])
            inside ^= cross
            j = i
    return inside


@st.cache_resource
def build_geometry():
    precip, gauges, delays, grid, neighbourhoods = load_data()
    cell_xy = to_xy(grid["latitude"], grid["longitude"])
    gauge_xy = to_xy(gauges["latitude"], gauges["longitude"])
    delay_xy = to_xy(delays["latitude"], delays["longitude"])

    cell_features = []
    for c in grid.itertuples(index=False):
        s, n = round(c.latitude - c.half_height_deg, 5), round(c.latitude + c.half_height_deg, 5)
        w, e = round(c.longitude - c.half_width_deg, 5), round(c.longitude + c.half_width_deg, 5)
        cell_features.append({"type": "Feature", "properties": {"neighbourhood": c.neighbourhood},
                              "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}})

    nb_features = [{"type": "Feature", "properties": {"neighbourhood": n["neighbourhood"]},
                    "geometry": {"type": "Polygon", "coordinates": n["rings"]}} for n in neighbourhoods]

    delay_nb = np.full(len(delays), "", dtype=object)
    lon, lat = delays["longitude"].to_numpy(), delays["latitude"].to_numpy()
    for n in neighbourhoods:
        hit = point_in_rings(lon, lat, n["rings"]) & (delay_nb == "")
        delay_nb[hit] = n["neighbourhood"]

    return {
        "cell_to_gauge": distance_matrix(cell_xy, gauge_xy),
        "delay_to_gauge": distance_matrix(delay_xy, gauge_xy),
        "cell_features": cell_features,
        "cell_neighbourhood": grid["neighbourhood"].to_numpy(),
        "nb_features": nb_features,
        "delay_neighbourhood": delay_nb,
    }


@st.cache_data
def delay_precip_estimates(winter_rule, power):
    precip, gauges, delays, _, _ = load_data()
    counted = apply_season_rule(precip, gauges, winter_rule)
    delay_to_gauge = build_geometry()["delay_to_gauge"]
    out = np.full(len(delays), np.nan)
    for day, idx in delays.groupby("date").indices.items():
        if day in counted.index:
            out[idx] = idw_estimate(delay_to_gauge[idx], counted.loc[day].to_numpy(), power)
    return np.round(out, 1)


# --------------------------------------------------------------------------------------
# Folium map
# --------------------------------------------------------------------------------------
def make_map(day, values, estimates, geo, gauges, day_delays, all_delays, fit_to_day, fixed_max, show_heat):
    import folium
    from branca.colormap import LinearColormap
    from folium.features import GeoJsonTooltip
    from folium.plugins import Fullscreen, HeatMap

    finite = estimates[np.isfinite(estimates)]
    if fit_to_day and finite.size:
        low, high = float(finite.min()), float(finite.max())
        caption = f"IDW daily precipitation (mm, scaled to this day: {low:.1f}–{high:.1f})"
    else:
        low, high = 0.0, float(fixed_max)
        caption = f"IDW daily precipitation (mm, fixed scale, darkest = {fixed_max:g} mm or more)"
    if high <= low:
        high = low + 0.1
    colours = LinearColormap(PRECIP_RAMP, vmin=low, vmax=high)
    colours.caption = caption

    def fill(v):
        return "#8c8c8c" if v is None or not np.isfinite(v) else colours(min(max(v, low), high))

    m = folium.Map(location=[43.72, -79.38], zoom_start=11, tiles="OpenStreetMap", control_scale=True)
    Fullscreen().add_to(m)

    # IDW surface
    cells = []
    for f, v in zip(geo["cell_features"], estimates):
        cells.append({**f, "properties": {**f["properties"], "precipitation_mm": None if not np.isfinite(v) else round(float(v), 1)}})
    folium.GeoJson(
        {"type": "FeatureCollection", "features": cells}, name="IDW precipitation surface",
        style_function=lambda f: {"fillColor": fill(f["properties"]["precipitation_mm"]), "color": None,
                                  "weight": 0, "fillOpacity": 0.65},
        tooltip=GeoJsonTooltip(fields=["neighbourhood", "precipitation_mm"], aliases=["Neighbourhood", "IDW precipitation (mm)"]),
    ).add_to(m)

    # Neighbourhood averages (off by default)
    means = pd.Series(estimates, index=geo["cell_neighbourhood"]).groupby(level=0).mean().round(1)
    counts = pd.Series(day_delays["neighbourhood"]).value_counts()
    nb = []
    for f in geo["nb_features"]:
        name = f["properties"]["neighbourhood"]
        mean = means.get(name, np.nan)
        nb.append({**f, "properties": {"neighbourhood": name,
                                       "precipitation_mm": None if pd.isna(mean) else float(mean),
                                       "delays": int(counts.get(name, 0))}})
    folium.GeoJson(
        {"type": "FeatureCollection", "features": nb}, name="Neighbourhood average precipitation", show=False,
        style_function=lambda f: {"fillColor": fill(f["properties"]["precipitation_mm"]), "color": "#444444",
                                  "weight": 1, "fillOpacity": 0.75},
        tooltip=GeoJsonTooltip(fields=["neighbourhood", "precipitation_mm", "delays"],
                               aliases=["Neighbourhood", "Average IDW precipitation (mm)", "Weather delays this day"]),
    ).add_to(m)
    folium.GeoJson({"type": "FeatureCollection", "features": geo["nb_features"]}, name="Neighbourhood outlines",
                   style_function=lambda f: {"fillOpacity": 0, "color": "#555555", "weight": 0.5}).add_to(m)

    # That day's delays
    layer = folium.FeatureGroup(name=f"Weather delays on {day.date()} ({len(day_delays)})").add_to(m)
    for r in day_delays.sort_values("delay_min", ascending=False).itertuples():
        precip_text = "n/a" if pd.isna(r.idw_precip_mm) else f"{r.idw_precip_mm:.1f} mm"
        popup = (f"<b>{escape(r.time)} · Route {escape(r.route)} {escape(r.direction)}</b><br>{escape(r.location)}<br>"
                 f"Delay {r.delay_min} min · gap {r.gap_min} min<br>IDW precipitation here: {precip_text}")
        folium.CircleMarker(
            location=[r.latitude, r.longitude], radius=float(3 + 11 * np.sqrt(min(r.delay_min, 600) / 600)),
            color="white", weight=1, fill=True, fill_color=DELAY_ORANGE, fill_opacity=0.85,
            tooltip=f"Route {r.route}: {r.delay_min} min", popup=folium.Popup(popup, max_width=260),
        ).add_to(layer)

    HeatMap(all_delays[["latitude", "longitude"]].to_numpy().tolist(), name="All weather delays (hot spots)",
            radius=12, blur=15, show=show_heat).add_to(m)

    gauge_layer = folium.FeatureGroup(name="Rain gauges").add_to(m)
    for g in gauges.itertuples(index=False):
        v = values.get(g.name, np.nan)
        state = f"{v:.1f} mm counted" if pd.notna(v) else "not counted / no readings that day"
        folium.RegularPolygonMarker(
            location=[g.latitude, g.longitude], number_of_sides=3, radius=6, rotation=30,
            color="#222222", weight=1, fill=True, fill_color="#222222" if pd.notna(v) else "#ffffff",
            fill_opacity=1, tooltip=f"{g.name} ({g.season}): {state}",
        ).add_to(gauge_layer)

    colours.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def show_map(m, height=620):
    """streamlit-folium if installed (interactive component), otherwise the map's own HTML."""
    try:
        from streamlit_folium import st_folium
        st_folium(m, height=height, use_container_width=True, returned_objects=[])
    except (ImportError, TypeError):
        import streamlit.components.v1 as components
        components.html(m.get_root().render(), height=height)


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
div[data-testid="stMetric"] { background: var(--secondary-background-color); border-radius: 8px; padding: 0.8rem 1rem; }
div[data-testid="stMetricValue"] { font-family: Georgia, serif; font-weight: 400; }
.rd-note { opacity: 0.7; font-size: 0.9rem; margin-top: -0.4rem; }
</style>""", unsafe_allow_html=True)

precip, gauges, all_delays, grid, _ = load_data()
first_day, last_day = precip.index.min().date(), precip.index.max().date()

with st.sidebar:
    st.header("Advanced settings")
    winter_rule = st.checkbox("Nov–Mar: count only the 4-season gauges", value=True,
                              help="Project assumption: the 3-season gauges are off in winter.")
    power = st.slider("IDW power (p)", 0.5, 5.0, 2.0, 0.5, help="Higher = nearby gauges count more.")
    fixed_max = st.slider("Fixed scale maximum (mm)", 5, 60, 25, 5)
    show_heat = st.checkbox("Show hot spots of all weather delays", value=False)

counted = apply_season_rule(precip, gauges, winter_rule)
summary = build_daily_summary(counted, all_delays)
delay_days = summary.index[summary["delays"] > 0]
groups = build_precip_groups(summary, all_delays)
geo = build_geometry()
delays = all_delays.assign(idw_precip_mm=delay_precip_estimates(winter_rule, power),
                           neighbourhood=geo["delay_neighbourhood"])


# ---- selected day (session state) -----------------------------------------------------
def set_day(d):
    d = min(max(pd.Timestamp(d).date(), first_day), last_day)
    st.session_state.day = d
    st.session_state.year = d.year


default_day = summary["delays"].idxmax().date()
remembered = st.session_state.get("day")
if not hasattr(remembered, "year") or not (first_day <= pd.Timestamp(remembered).date() <= last_day):
    set_day(default_day)
if "year" not in st.session_state:
    st.session_state.year = st.session_state.day.year


def step_delay_day(direction):
    current = pd.Timestamp(st.session_state.day)
    options = delay_days[delay_days > current] if direction > 0 else delay_days[delay_days < current]
    if len(options):
        set_day(options[0] if direction > 0 else options[-1])


def reset_view():
    set_day(default_day)
    st.session_state.scale = "Fixed scale"


def on_date_change():
    st.session_state.year = st.session_state.day.year


# ---- header ---------------------------------------------------------------------------
st.title("Precipitation and TTC weather-related bus delays")
in_record = summary.loc[all_delays["date"].min():all_delays["date"].max()].dropna(subset=["mean_precip_mm"])
dry = in_record.loc[in_record["mean_precip_mm"] < 0.2, "delays"].mean()
wet = in_record.loc[in_record["mean_precip_mm"] >= 10, "delays"].mean()
winter_share = all_delays["date"].dt.month.isin(WINTER_MONTHS).mean()
st.markdown(
    f"Weather-related bus delays rise with precipitation: **{wet:.2f}** delays a day on days averaging 10 mm or more, "
    f"against **{dry:.2f}** on dry days. **{winter_share:.1%}** of the delays fall between November and March, "
    f"when the totals are mostly snow measured by the five 4-season gauges."
)

# ---- controls -------------------------------------------------------------------------
try:
    cols = st.columns([2, 1.3, 1.3, 2.2, 0.8], vertical_alignment="bottom")
except TypeError:  # Streamlit older than 1.36
    cols = st.columns([2, 1.3, 1.3, 2.2, 0.8])
c_date, c_prev, c_next, c_scale, c_reset = cols
c_date.date_input("Date", key="day", min_value=first_day, max_value=last_day, on_change=on_date_change)
c_prev.button("← Previous delay day", on_click=step_delay_day, args=(-1,), use_container_width=True)
c_next.button("Next delay day →", on_click=step_delay_day, args=(1,), use_container_width=True)
scale = c_scale.radio("Map colour scale", ["Fixed scale", "Fit to day"], horizontal=True, key="scale")
c_reset.button("Reset", on_click=reset_view, use_container_width=True)

day = pd.Timestamp(st.session_state.day)
values = counted.loc[day] if day in counted.index else pd.Series(np.nan, index=gauges["name"])
estimates = idw_estimate(geo["cell_to_gauge"], values.to_numpy(), power)
day_delays = delays[delays["date"] == day]
row = summary.loc[day] if day in summary.index else None
no_precip = row is None or pd.isna(row["mean_precip_mm"])

# ---- selected-day tiles ---------------------------------------------------------------
st.markdown(f'<p class="rd-note">Showing <b>{day:%A, %B} {day.day}, {day.year}</b></p>', unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Weather delays", f"{len(day_delays):,}", help="Incidents logged that day")
m2.metric("Delay minutes", f"{int(day_delays['delay_min'].sum()):,}", help="Sum of reported delays")
m3.metric("City average precipitation", "—" if no_precip else f"{row['mean_precip_mm']:.1f} mm", help="Mean of counted gauges")
m4.metric("Wettest gauge", "—" if no_precip else f"{row['max_precip_mm']:.1f} mm",
          None if no_precip else str(row["wettest_gauge"]), delta_color="off")
m5.metric("Gauges counted", "—" if row is None else f"{int(row['gauges_counted'])}", help="Gauges with readings that day")

# ---- map ------------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("#### Precipitation surface and delay locations")
    st.markdown('<p class="rd-note">Inverse distance weighted (IDW) daily precipitation on a grid over the city, with that '
                "day's weather-related delays as circles sized by delay minutes. Click a circle for details; use the "
                "layer box to switch layers.</p>", unsafe_allow_html=True)
    show_map(make_map(day, values, estimates, geo, gauges, day_delays, all_delays,
                      scale == "Fit to day", fixed_max, show_heat))


# ---- timeline -------------------------------------------------------------------------
def pick_from_timeline():
    event = st.session_state.get("timeline")
    selection = getattr(event, "selection", None) or {}
    picks = list(selection.get("pick_precip") or []) + list(selection.get("pick_delay") or [])
    if picks and "date" in picks[-1]:
        v = picks[-1]["date"]
        set_day(pd.to_datetime(v, unit="ms") if isinstance(v, (int, float)) else pd.to_datetime(v))


with st.container(border=True):
    st.markdown("#### Daily precipitation and delays through the year")
    st.markdown('<p class="rd-note">City average precipitation (top) and weather-related delays (bottom) for each day of '
                "the selected year. Click a day to show it on the map.</p>", unsafe_allow_html=True)
    st.selectbox("Year", sorted({d.year for d in summary.index}), key="year", label_visibility="collapsed")
    year_rows = summary.loc[str(st.session_state.year)].reset_index()
    tooltip = [alt.Tooltip("date:T", title="Day", format="%a %b %d, %Y"),
               alt.Tooltip("mean_precip_mm:Q", title="Avg precipitation (mm)", format=".1f"),
               alt.Tooltip("delays:Q", title="Delays"),
               alt.Tooltip("delay_minutes:Q", title="Delay minutes", format=",")]
    pick_precip = alt.selection_point(name="pick_precip", fields=["date"], on="click")
    pick_delay = alt.selection_point(name="pick_delay", fields=["date"], on="click")
    base = alt.Chart(year_rows).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%b", tickCount="month")))
    marker = alt.Chart(pd.DataFrame({"date": [day]})).mark_rule(color="#888", strokeDash=[4, 3]).encode(x="date:T")
    precip_chart = (base.mark_bar(color=PRECIP_BLUE).encode(y=alt.Y("mean_precip_mm:Q", title="Precipitation, mm"), tooltip=tooltip)
                  .add_params(pick_precip).properties(height=160))
    delay_chart = (base.mark_bar(color=DELAY_ORANGE).encode(y=alt.Y("delays:Q", title="Delays"), tooltip=tooltip)
                   .add_params(pick_delay).properties(height=100))
    timeline = alt.vconcat(precip_chart + marker, delay_chart + marker, spacing=6).resolve_scale(x="shared")
    try:
        st.altair_chart(timeline, use_container_width=True, key="timeline", on_select=pick_from_timeline)
    except TypeError:  # Streamlit older than 1.35: chart shows, clicks don't select
        st.altair_chart(timeline, use_container_width=True)

# ---- precipitation groups and delay days ---------------------------------------------------
top_days = summary[summary["delays"] > 0].sort_values("delays", ascending=False).reset_index()
top_days_table = pd.DataFrame({
    "Date": top_days["date"].dt.date, "Delays": top_days["delays"], "Delay min": top_days["delay_minutes"],
    "Avg precipitation (mm)": top_days["mean_precip_mm"], "Routes": top_days["routes"],
})


def pick_from_table():
    event = st.session_state.get("top_days")
    selection = getattr(event, "selection", None)
    rows = []
    if selection is not None:
        rows = list(selection.get("rows", [])) if isinstance(selection, dict) else list(getattr(selection, "rows", []))
    if rows:
        set_day(top_days_table.iloc[rows[0]]["Date"])


left, right = st.columns(2)
with left.container(border=True):
    st.markdown("#### Delays per day by precipitation")
    st.markdown('<p class="rd-note">Average weather-related delays per day, grouping every day in the delay record '
                "by its city average precipitation.</p>", unsafe_allow_html=True)
    labelled = groups.assign(label=groups["delays_per_day"].map("{:.2f} / day".format) + " · "
                             + groups["days"].map("{:,} days".format))
    y = alt.Y("precip_group:N", sort=PRECIP_GROUP_LABELS, title=None)
    bars = alt.Chart(labelled).mark_bar(color=DELAY_ORANGE, cornerRadiusEnd=3).encode(
        y=y, x=alt.X("delays_per_day:Q", title="Weather delays per day", scale=alt.Scale(nice=True)),
        tooltip=[alt.Tooltip("precip_group:N", title="City average precipitation"), alt.Tooltip("days:Q", title="Days", format=","),
                 alt.Tooltip("delays:Q", title="Delays", format=","),
                 alt.Tooltip("delays_per_day:Q", title="Per day", format=".2f")])
    labels = alt.Chart(labelled).mark_text(align="left", dx=4).encode(y=y, x="delays_per_day:Q", text="label:N")
    st.altair_chart((bars + labels).properties(height=280, padding={"right": 110}), use_container_width=True)

with right.container(border=True):
    st.markdown("#### Days with weather delays")
    st.markdown('<p class="rd-note">Most delays first. Click a row to show that day.</p>', unsafe_allow_html=True)
    try:
        st.dataframe(top_days_table, hide_index=True, use_container_width=True, height=300, key="top_days",
                     on_select=pick_from_table, selection_mode="single-row")
    except TypeError:  # Streamlit older than 1.35
        st.dataframe(top_days_table, hide_index=True, use_container_width=True, height=300)

# ---- delays on the selected day -------------------------------------------------------
with st.container(border=True):
    st.markdown("#### Delays on the selected day")
    st.markdown('<p class="rd-note">Each weather-related delay with the IDW precipitation estimated at its location.</p>',
                unsafe_allow_html=True)
    if len(day_delays):
        table = day_delays[["time", "route", "location", "direction", "delay_min", "gap_min", "idw_precip_mm"]].rename(
            columns={"time": "Time", "route": "Route", "location": "Location", "direction": "Dir",
                     "delay_min": "Delay (min)", "gap_min": "Gap (min)", "idw_precip_mm": "IDW precipitation (mm)"})
        st.dataframe(table, hide_index=True, use_container_width=True, height=360)
        st.download_button("Download this day's delays (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name=f"weather_delays_{day.date()}.csv", mime="text/csv")
    else:
        st.caption("No weather-related delays were logged on this day. Use Previous / Next delay day to jump to one.")

# ---- notes ----------------------------------------------------------------------------
with st.expander("How the numbers are made"):
    gauge_rule = ("April to October uses every gauge. November to March uses only the five 4-season gauges, following "
                  "the project's assumption that the 3-season gauges are switched off then; winter totals are treated "
                  "as snow." if winter_rule else
                  "Every gauge, all year (the winter rule is switched off in Advanced settings).")
    st.markdown(f"""
**Daily precipitation** — each City of Toronto gauge's 5-minute readings (Toronto Open Data) summed by calendar day in mm,
after removing missing, negative and duplicate readings.

**Which gauges count** — {gauge_rule}

**IDW precipitation surface** — each grid cell (about 500 m) and each delay location gets a weighted average of the counted
gauges that reported that day, with weight 1 / distance^{power:g}, so nearby gauges count most. Gauges with no readings
are left out rather than counted as 0 mm.

**City average precipitation** — the plain mean of the counted gauges' daily totals; it is what the tiles, the timeline and
the precipitation groups use.

**Delays** — {len(all_delays):,} TTC bus delays recorded as weather related, located by the coordinates in the delay
file. Coverage is uneven: most records are from 2024–2026, so earlier years show fewer delays.

**Left out** — delay days before the rain gauge record starts ({first_day:%B %Y}) have no precipitation and appear only in
the delay table. The charts show that delays and precipitation go together; on their own they don't prove precipitation causes them.

Open the sidebar (› at the top left) for the Bus delay EDA page and advanced settings: the winter gauge rule, IDW power, fixed-scale maximum and a
hot-spot layer of all weather delays.
""")
