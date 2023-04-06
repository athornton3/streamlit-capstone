# app.py

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import re
import s3fs 
import pickle

# Create connection object.
# `anon=False` means not anonymous, i.e. it uses access keys to pull data.
fs = s3fs.S3FileSystem(anon=False)

#create data list to feed to model
#project_texts = projects_df['AwardTitle'].astype(str) + '[SEP]' + projects_df['AbstractNarration'].astype(str)
	
st.set_page_config(
	layout="wide",
	page_title="Research Matching",
	page_icon="🐙",
)


st.title("Find Related Research Info 🐙")

with st.expander("ℹ️ - About this app", expanded=False):
    st.write(
        """ -   This app uses embedding created with SPECTER and awarded NSF grants from 2021-2023. Paste and get matching projects and more info """
    )
    st.markdown("")

#Create input form for title and abstract 
with st.form("form1", clear_on_submit=False): 
    c1, c2 = st.columns([1,2], gap="small")
    with c1:
        title = st.text_input("Title") 
        abstract = st.text_area("Abstract (max 300 words)",
                height=300) 
    
        #check for max words
        MAX_WORDS = 300
        res = len(re.findall(r"\w+", abstract))
        if res > MAX_WORDS:
            st.warning("⚠️ Your text contains " + str(res) + " words." + " Only the first 300 words will be reviewed.")
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
def read_file(filename):
	with fs.open(filename, encoding='utf-8') as f:
		df = pd.read_csv(f)
	return df 
@st.cache_data
def get_embeddings(filename):
	with fs.open(filename, 'rb') as f:
		df = pickle.loads(f)
	return df 
content = read_file("streamlitbucketcapstoneajt/export_21_22_23_col_rv_100_latlong.csv")
papers_df = content[['latitude','longitude']].dropna()
embeddings = get_embeddings("streamlitbucketcapstoneajt/corpus_embeddings.pickle")

try:
    #df = get_UN_data()
    st.dataframe(content)
    with c2: # Map demo
    	map_data = papers_df

    	st.subheader('Matching Research Institutions')
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
