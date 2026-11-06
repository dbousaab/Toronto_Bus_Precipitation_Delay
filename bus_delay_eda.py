"""Bus delay EDA (pre-2025): incident reasons and delay hot spots, from pre_2025_ttc_bus_EDA.ipynb."""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
REASONS_FILE = DATA / "bus_incident_reasons_by_year.csv"
COORDS_FILE = DATA / "bus_focus_delays_with_coords.csv"
FOCUS_REASONS = ["General Delay", "Operations", "Diversion", "Other"]
FOCUS_COLOUR = "#e8590c"
OTHER_COLOUR = "#9aa5b1"

st.markdown("""
<style>
.block-container { padding-top: 2rem; }
div[data-testid="stMetric"] { background: var(--secondary-background-color); border-radius: 8px; padding: 0.8rem 1rem; }
div[data-testid="stMetricValue"] { font-family: Georgia, serif; font-weight: 400; }
.rd-note { opacity: 0.7; font-size: 0.9rem; margin-top: -0.4rem; }
</style>""", unsafe_allow_html=True)

st.title("TTC bus delays before 2025: exploratory analysis")

if not REASONS_FILE.exists() or not COORDS_FILE.exists():
    st.info(
        "This page needs two small data files that are made from the pre-2025 TTC bus delay CSVs.\n\n"
        "1. Put `pre_2025_ttc_bus_delays_cleaned.csv` and `pre_2025_ttc_bus_delays_with_coords.csv` "
        "in the same folder as this app.\n"
        "2. Run `python prepare_bus_eda_data.py` there. It writes `data/bus_incident_reasons_by_year.csv` "
        "and `data/bus_focus_delays_with_coords.csv`.\n"
        "3. Upload those two files to the `data` folder in GitHub (the big CSVs don't need to be uploaded)."
    )
    st.stop()


@st.cache_data
def load_eda():
    reasons = pd.read_csv(REASONS_FILE)
    points = pd.read_csv(COORDS_FILE, dtype={"route": str, "location": str, "located_by": str})
    points["date"] = pd.to_datetime(points["date"])
    for col in ("route", "location", "located_by"):
        if col in points:
            points[col] = points[col].fillna("")
    return reasons, points


reasons, points = load_eda()
years = sorted(reasons["year"].unique())
focus_present = [r for r in FOCUS_REASONS if r in set(reasons["incident"])]
focus_names = ", ".join(focus_present[:-1]) + (" and " if len(focus_present) > 1 else "") + focus_present[-1]

st.markdown(
    f"TTC bus delays recorded {years[0]}–{years[-1]}, by incident reason. The project's focus reasons "
    f"(**{focus_names}**) are highlighted; the map shows where those delays happen."
)

# ---- filters ------------------------------------------------------------------------------
year_range = st.select_slider("Years", options=years, value=(years[0], years[-1]))
in_years = reasons[reasons["year"].between(*year_range)]
by_reason = (in_years.groupby("incident")[["delays", "delay_minutes"]].sum()
             .sort_values("delays", ascending=False).reset_index())
by_reason["focus"] = np.where(by_reason["incident"].isin(focus_present), "Focus reason", "Other reasons")
by_reason["share"] = by_reason["delays"] / by_reason["delays"].sum()
focus = by_reason[by_reason["incident"].isin(focus_present)]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Bus delays", f"{by_reason['delays'].sum():,}", help=f"All reasons, {year_range[0]}–{year_range[1]}")
k2.metric("Incident reasons", f"{len(by_reason)}")
k3.metric("Focus-reason delays", f"{focus['delays'].sum():,}", f"{focus['share'].sum():.1%} of all", delta_color="off")
top = by_reason.iloc[0]
k4.metric("Most common reason", top["incident"], f"{top['share']:.1%} of delays", delta_color="off")

# ---- incident reason charts -----------------------------------------------------------------
left, right = st.columns([3, 2])
with left.container(border=True):
    st.markdown("#### Distribution of incident reasons")
    st.markdown('<p class="rd-note">Every bus delay in the selected years; focus reasons in orange.</p>',
                unsafe_allow_html=True)
    chart = alt.Chart(by_reason).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("incident:N", sort="-x", title=None),
        x=alt.X("delays:Q", title="Delays"),
        color=alt.Color("focus:N", scale=alt.Scale(domain=["Focus reason", "Other reasons"],
                                                    range=[FOCUS_COLOUR, OTHER_COLOUR]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("incident:N", title="Reason"), alt.Tooltip("delays:Q", title="Delays", format=","),
                 alt.Tooltip("share:Q", title="Share", format=".1%"),
                 alt.Tooltip("delay_minutes:Q", title="Delay minutes", format=",")],
    ).properties(height=max(260, 24 * len(by_reason)))
    st.altair_chart(chart, use_container_width=True)

with right.container(border=True):
    st.markdown("#### Focus reasons")
    st.markdown('<p class="rd-note">The reasons kept for the analysis.</p>', unsafe_allow_html=True)
    fchart = alt.Chart(focus).mark_bar(color=FOCUS_COLOUR, cornerRadiusEnd=3).encode(
        x=alt.X("incident:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("delays:Q", title="Delays"),
        tooltip=[alt.Tooltip("incident:N", title="Reason"), alt.Tooltip("delays:Q", title="Delays", format=","),
                 alt.Tooltip("delay_minutes:Q", title="Delay minutes", format=",")],
    ).properties(height=300)
    st.altair_chart(fchart, use_container_width=True)

    trend = in_years[in_years["incident"].isin(focus_present)]
    if trend["year"].nunique() > 1:
        st.markdown("##### Focus reasons by year")
        st.altair_chart(alt.Chart(trend).mark_line(point=True).encode(
            x=alt.X("year:O", title=None), y=alt.Y("delays:Q", title="Delays"),
            color=alt.Color("incident:N", title=None, legend=alt.Legend(orient="bottom")),
            tooltip=["year:O", "incident:N", alt.Tooltip("delays:Q", format=",")],
        ).properties(height=200), use_container_width=True)

# ---- hot-spot map ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("#### Where focus-reason delays happen")
    st.markdown('<p class="rd-note">Heat map of delays with known coordinates. Pick one day (as in the notebook) or '
                "show every day in the selected years.</p>", unsafe_allow_html=True)
    pts = points[points["date"].dt.year.between(*year_range)]
    c1, c2, c3 = st.columns([2, 2, 3])
    mode = c1.radio("Show", ["One day", "All days"], horizontal=True)
    days = pts["date"].dt.date.drop_duplicates().sort_values()
    if mode == "One day" and len(days):
        busiest = pts.groupby(pts["date"].dt.date).size().idxmax()
        chosen = c2.date_input("Day", value=busiest, min_value=days.iloc[0], max_value=days.iloc[-1])
        pts = pts[pts["date"].dt.date == chosen]
    chosen_reasons = c3.multiselect("Reasons", focus_present, default=focus_present)
    pts = pts[pts["incident"].isin(chosen_reasons)]

    import folium
    from folium.plugins import Fullscreen, HeatMap

    m = folium.Map(location=[43.70, -79.40], zoom_start=11, tiles="OpenStreetMap", control_scale=True)
    Fullscreen().add_to(m)
    heat_points = pts[["latitude", "longitude"]]
    if len(heat_points) > 60000:  # keep the map responsive
        heat_points = heat_points.sample(60000, random_state=0)
    if len(heat_points):
        HeatMap(heat_points.to_numpy().tolist(), radius=15 if mode == "One day" else 9,
                blur=15 if mode == "One day" else 11, name="Delay hot spots").add_to(m)
    if mode == "One day" and 0 < len(pts) <= 400:
        dots = folium.FeatureGroup(name="Individual delays", show=False).add_to(m)
        for r in pts.itertuples():
            folium.CircleMarker([r.latitude, r.longitude], radius=4, color=FOCUS_COLOUR, fill=True, fill_opacity=0.8,
                                tooltip=f"{r.time} · {r.incident} · Route {r.route} · {r.delay_min} min · {r.location}"
                                        + (f" (located by {r.located_by})" if getattr(r, "located_by", "") else "")
                                ).add_to(dots)
    folium.LayerControl(collapsed=True).add_to(m)
    st.caption(f"{len(pts):,} delays shown" + (" (map uses a 60,000-point sample)" if len(pts) > 60000 else ""))
    try:
        from streamlit_folium import st_folium
        st_folium(m, height=560, use_container_width=True, returned_objects=[])
    except (ImportError, TypeError):
        import streamlit.components.v1 as components
        components.html(m.get_root().render(), height=560)

with st.expander("About this page"):
    st.markdown("""
- **Source:** the TTC bus delay records before 2025 used in `pre_2025_ttc_bus_EDA.ipynb`
  (`pre_2025_ttc_bus_delays_with_coords.csv`): every delay that could be given a location.
- **Focus reasons:** the reasons the notebook keeps (General Delay, Operations, Diversion and Other);
  "Other" does not occur in this file, so only the reasons present are shown.
- **Map locations:** each delay is placed at its station, intersection or route midpoint ("located by" in the
  point tooltips), so route-midpoint points show where a route runs rather than the exact incident spot.
  Points outside the Toronto area are dropped.
""")
