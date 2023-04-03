# app.py

import streamlit as st
import numpy as np
import pandas as pd

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

st.text_input("Your name", key="name")

# You can access the value at any point with:
st.session_state.name
