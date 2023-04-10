# app.py

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import re
import s3fs 
import pickle
#from datasets import Dataset
import sentence_transformers
from sentence_transformers import SentenceTransformer, util
import leafmap.foliumap as leafmap
from streamlit_folium import folium_static

# Create connection object.
# `anon=False` means not anonymous, i.e. it uses access keys to pull data.
fs = s3fs.S3FileSystem(anon=False)
	
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
def load_model(filename):
	return SentenceTransformer('allenai-specter')

@st.cache_data
def get_embeddings(_model, data):
	embeddings = model.encode(data, convert_to_tensor=True)
	return embeddings

@st.cache_data
def read_embeddings(filename):
    with fs.open(filename, 'rb') as pkl:
        cache_data = pickle.loads(pkl.read())
    return cache_data

projects_df = read_file("streamlitbucketcapstoneajt/export_21_22_23_col_rv_100_latlong.csv")
project_texts = projects_df['AwardTitle'].astype(str) + '[SEP]' + projects_df['AbstractNarration'].astype(str)

embeddings = read_embeddings("streamlitbucketcapstoneajt/corpus_embeddings.pkl")
#embeddings = get_embeddings(model, project_texts) #only if embeddings not found/not available

model = load_model()

#function to take title & abstract and search corpus for similar projects
def search_projects(title, abstract, n):
    query_embedding = model.encode(title+'[SEP]'+abstract, convert_to_tensor=True)

    results = util.semantic_search(query_embedding, embeddings, top_k = n)
    results_normalized = util.semantic_search(query_embedding, embeddings, score_function=util.dot_score, top_k = n)
    df = pd.DataFrame()
    scores = []
    rows = []
    for prj in results[0]:
        related_project = projects_df.loc[prj['corpus_id']]
        #rows.append(projects_df.loc[prj['corpus_id']])
        scores.append(prj['score'])
        df = df.append(related_project)
    df.insert(0, "cosim_score", scores)
    #df = pd.concat(rows, scores)
    return df

try:
    df = search_projects(title,abstract, numResults)
    st.dataframe(df)
    with c2: # Map demo
        matches_df = df[['latitude','longitude', 'Institution-Name']].dropna()
        map_data = matches_df
        st.subheader('Matching Research Institutions')
        #st.map(map_data)
        m = leafmap.Map(center=(39.381266, -97.922211), zoom=4)
        m.add_circle_markers_from_xy(map_data, x="longitude", y="latitude", radius=5, tooltip="latitude", popup='Institution-Name') #min-width and max-width for the popup
        #add_text and add_legend https://leafmap.org/foliumap/#leafmap.foliumap.Map.add_legend
        folium_static(m)
		
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
