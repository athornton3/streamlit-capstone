# app.py

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

# Map demo
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.subheader('Map Title Here')
st.map(map_data)

#text box demo
add_textbox = st.sidebar.text_input("Research Project Title", key="title")
# You can access the value at any point with:
st.session_state.title

add_textbox2 = st.sidebar.text_input("Research Project Abstract", key="abstract")
st.session_state.abstract

#CREATING OUR FORM FIELDS 
with st.form("form1", clear_on_submit=True): 
    name = st.text_input("Enter full name") 
    email = st.text_input("Enter email") 
    message = st.text_area("Message") 
    age = st.slider("Enter your age", min_value = 10, max_value = 100) 
    st.write(age)

    submit = st.form_submit_button("Submit this form")

# data fetch and use check  instructions here https://docs.streamlit.io/knowledge-base/tutorials/databases/aws-s3
@st.cache_data
def get_UN_data():
    AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
    df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
    return df.set_index("Region")

try:
    df = get_UN_data()
    countries = st.multiselect(
        "Choose countries", list(df.index), ["China", "United States of America"]
    )
    if not countries:
        st.error("Please select at least one country.")
    else:
        data = df.loc[countries]
        data /= 1000000.0
        st.write("### Gross Agricultural Production ($B)", data.sort_index())

        data = data.T.reset_index()
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
        )
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                color="Region:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )
# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
