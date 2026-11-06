"""
TTC bus delays and precipitation - Streamlit app (two pages).

Local run:  pip install -r requirements.txt
            streamlit run streamlit_app.py
Deploy:     push this folder to GitHub and create the app on https://share.streamlit.io
            with main file "streamlit_app.py".
"""
import streamlit as st

st.set_page_config(page_title="Precipitation and TTC bus delays", page_icon="🌧️", layout="wide")

pages = st.navigation([
    st.Page("precipitation_delays.py", title="Precipitation & weather delays", icon="🌧️", default=True),
    st.Page("bus_delay_eda.py", title="Bus delay EDA (pre-2025)", icon="🚌"),
])
pages.run()
