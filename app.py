# app.py

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Side Bar 🎈")

#text box demo
add_textbox = st.sidebar.text_input("Research Project Title", key="title")
# You can access the value at any point with:
st.session_state.title

add_textbox2 = st.sidebar.text_input("Research Project Abstract", key="abstract")
st.session_state.abstract

#Create input form for title and abstract 
with st.form("form1", clear_on_submit=False): 
    title = st.text_input("Title") 
    abstract = st.text_area("Abstract (max 300 words)",
            height=300) 
    
    #check for max words
    MAX_WORDS = 300
    import re
    res = len(re.findall(r"\w+", abstract))
    if res > MAX_WORDS:
        st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 300 words will be reviewed."
            )

        abstract = abstract[:MAX_WORDS]
    
    numResults = st.slider(
            "Number of results",
            min_value=3,
            max_value=30,
            value=10,
            help="You can choose the number of results to display. Between 3 and 30, default number is 10.",
        )

    submit_button = st.form_submit_button("Submit title/abstract")

if not submit_button:
    st.stop()

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
    # Map demo
    map_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon'])

    st.subheader('Map Title Here')
    st.map(map_data)

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
