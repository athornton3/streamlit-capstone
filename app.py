# app.py

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import re
import s3fs 
import boto3
import io
import pickle
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

st.title("Find Related Research Info 🔍")

with st.expander("ℹ️ - About this app - DS785 Capstone SP23SEC01", expanded=False):
    st.write(
        """ -   This app uses embedding created with SPECTER and awarded NSF grants from 2023. Paste and get matching projects and more info """
    )
    st.markdown("")

#Create input form for title and abstract 
with st.form("form1", clear_on_submit=False): 
    c1, c2 = st.columns([1,2], gap="large")
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
        
        includeConf = st.checkbox('Include Conference Awards', value=True)
        
        numResults = st.slider(
            "Max number of results",
            min_value=3,
            max_value=50,
            value=10,
            help="You can choose the number of results to display. Between 3 and 30, default number is 10.",
            )
        
        #dollarThreshold = st.slider(
        #    "Awarded Amount",
        #    min_value=0,
        #    max_value=10000000,
        #    value=0,
        #    help="You can choose the minimum dollar threshold for results. Between $0 and $10million, default is $0.",
        #    )
            
        
        submit_button = st.form_submit_button("Submit title/abstract")

if not submit_button:
    st.stop()

# data fetch from s3
@st.cache_data
def read_file(filename):
	with fs.open(filename, encoding='utf-8') as f:
		df = pd.read_csv(f)
	return df 

@st.cache_data
def load_model(tf):
	return SentenceTransformer(tf)

# read stored embeddings fom s3
@st.cache_data
def read_embeddings(filename):
    with fs.open(filename, 'rb') as pkl:
        cache_data = pickle.loads(pkl.read())
    return cache_data

#projects_df = read_file("streamlitbucketcapstoneajt/export_2023_col_rv_latlong_narrow_topics.csv")
projects_df = pd.read_csv("export_2023_col_rv_latlong_narrow_topics.csv")
project_texts = projects_df['AwardTitle'].astype(str) + '[SEP]' + projects_df['AbstractNarration'].astype(str)
model = load_model('allenai-specter')

#embeddings = read_embeddings("streamlitbucketcapstoneajt/corpus_embeddings_2023.pkl")
@st.cache_data
def get_embeddings(_model, data):
	embeddings = model.encode(data, convert_to_tensor=True)
	return embeddings
embeddings = get_embeddings(model, project_texts) #only if embeddings not found/not available
#pickle.dump(embeddings, fs.open(f"streamlitbucketcapstoneajt/corpus_embeddings_2023.pkl",'wb'))
#function to take title & abstract and search corpus for similar projects
def search_projects(title, abstract, n):
    query_embedding = model.encode(title+'[SEP]'+abstract, convert_to_tensor=True)

    results = util.semantic_search(query_embedding, embeddings, top_k = n)
    #results_normalized = util.semantic_search(query_embedding, embeddings, score_function=util.dot_score, top_k = n)
    df = pd.DataFrame()
    scores = pd.DataFrame()
    for prj in results[0]:
        related_project = projects_df.loc[projects_df["Unnamed: 0"] == prj['corpus_id']]
        scores = scores.append({"Unnamed: 0" : prj['corpus_id'], "cosim" : prj['score']}, ignore_index=True)
        df = df.append(related_project) #deprecated but couldn't get pd.concat to work
    #st.dataframe(df)  
    df2 = scores.merge(df, on="Unnamed: 0")  
    #st.write(scores)
    return df2

try:
    df = search_projects(title, abstract, numResults)
    #st.dataframe(df)
    display_df = df[['Unnamed: 0', 'cosim', 'Name', 'AwardTitle', 'AbstractNarration', 'AwardID', 'AwardAmount', 'AwardEffectiveDate', 'Organization-Directorate-LongName','Organization-Division-LongName','Institution-Name','Investigator-PI_FULL_NAME', 'ProgramElement-Text']]
    if includeConf:
        new_df = display_df
    else:
        new_df = display_df[~display_df["AbstractNarration"].str.contains('conference', na=False)]
        
    new_df.rename(columns = {'Unnamed: 0': "Index", 'cosim':'Similarity', 'Name':'Topic', 'AwardTitle':
    'Award Title', 'AbstractNarration':'Abstract', 'AwardAmount':'Award Amt', 'Organization-Directorate-LongName':'NSF Directorate','Organization-Division-LongName':'NSF Division','ProgramElement-Text': 'Funding Program', 'Institution-Name':'Institution','Investigator-PI_FULL_NAME':'PI'}, inplace = True) 
    st.dataframe(new_df)
    with c2: # Map demo
        map_data = df[['latitude','longitude','Institution-Name','Investigator-PI_FULL_NAME']]
        map_data['Institution'] = map_data['Institution-Name'] + '<br/><b>PI:</b> ' + map_data['Investigator-PI_FULL_NAME']
        map_data = map_data[['latitude','longitude','Institution']].dropna()
        st.subheader('Matching Research Institutions')
	
        m = leafmap.Map(center=(39.381266, -97.922211), zoom=4)
        m.add_circle_markers_from_xy(map_data, x="longitude", y="latitude", radius=5, popup='Institution')
        m.add_text(text="Map of Institutions", position="bottomright")
        
        folium_static(m)
		
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )

#st.button("Re-run")
