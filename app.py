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

# data fetch from s3
@st.cache_data
def read_file(filename):
	with fs.open(filename, encoding='utf-8') as f:
		df = pd.read_csv(f)
	return df 

@st.cache_data
def load_model(tf):
	return SentenceTransformer(tf)

# create embeddings - slow and resource intensive
@st.cache_data
def get_embeddings(_model, data):
	embeddings = model.encode(data, convert_to_tensor=True)
	return embeddings

# read stored embeddings fom s3
@st.cache_data
def read_embeddings(filename):
    with fs.open(filename, 'rb') as pkl:
        cache_data = pickle.loads(pkl.read())
    return cache_data

projects_df = read_file("streamlitbucketcapstoneajt/export_21_22_23_col_rv_100_latlong.csv")
project_texts = projects_df['AwardTitle'].astype(str) + '[SEP]' + projects_df['AbstractNarration'].astype(str)

embeddings = read_embeddings("streamlitbucketcapstoneajt/corpus_embeddings.pkl")
#embeddings = get_embeddings(model, project_texts) #only if embeddings not found/not available

model = load_model('allenai-specter')

s3 = boto3.client('s3')

#select column headers
resp = s3.select_object_content(
    Bucket='streamlitbucketcapstoneajt',
    Key='export_21_22_23_col_rv_100_latlong.csv',
    ExpressionType='SQL',
    Expression="SELECT * FROM s3object s limit 1",
    InputSerialization = {'CSV': {"FileHeaderInfo": "None"}, 'CompressionType': 'NONE'},
    OutputSerialization = {'CSV': {}},
)
for event in resp['Payload']:
	if 'Records' in event:
		records = event['Records']['Payload'].decode('utf-8')
		columns = pd.read_csv(io.StringIO(records), sep=",")
		st.write(columns.columns)
	elif 'Stats' in event:
		statsDetails = event['Stats']['Details']
		st.write("Stats details bytesScanned: "+str(statsDetails['BytesScanned']))


resp = s3.select_object_content(
    Bucket='streamlitbucketcapstoneajt',
    Key='export_21_22_23_col_rv_100_latlong.csv',
    ExpressionType='SQL',
    Expression="SELECT * FROM s3object s where s.\"AwardTitle\" = 'CI CoE: Demo Pilot: Advancing Research Computing and Data: Strategic Tools, Practices, and Professional Development'",
    InputSerialization = {'CSV': {"FileHeaderInfo": "Use"}, 'CompressionType': 'NONE'},
    OutputSerialization = {'CSV': {}},
)

for event in resp['Payload']:
	if 'Records' in event:
		records = event['Records']['Payload'].decode('utf-8')
		df = pd.read_csv(io.StringIO(records), sep=",")
		df = df.append(dict(zip(columns.columns, [col for col in df])))
		#df.append(dict(zip(df.columns, [col for col in df])), ignore_index=True)
		#hold = columns.columns.to_list()
		#hold = hold.insert(0,"test")
		#df.columns = columns.columns.to_list()
		#df = pd.concat([df, list],ignore_index = True)
		st.dataframe(df)
	elif 'Stats' in event:
		statsDetails = event['Stats']['Details']
		st.write("Stats details bytesScanned: "+str(statsDetails['BytesScanned']))

			
#function to take title & abstract and search corpus for similar projects
def search_projects(title, abstract, n):
    query_embedding = model.encode(title+'[SEP]'+abstract, convert_to_tensor=True)

    results = util.semantic_search(query_embedding, embeddings, top_k = n)
    results_normalized = util.semantic_search(query_embedding, embeddings, score_function=util.dot_score, top_k = n)
    df = pd.DataFrame()
    scores = []
    for prj in results[0]:
        related_project = projects_df.loc[prj['corpus_id']]
        scores.append(prj['score'])
        df = df.append(related_project) #deprecated but couldn't get pd.concat to work
    df.insert(0, "cosim_score", scores)
    return df

def search_projects_sql(title, abstract, n):
    query_embedding = model.encode(title+'[SEP]'+abstract, convert_to_tensor=True)

    results = util.semantic_search(query_embedding, embeddings, top_k = n)
    results_normalized = util.semantic_search(query_embedding, embeddings, score_function=util.dot_score, top_k = n)
    df = pd.DataFrame()
    scores = []
    #for prj in results[0]:
    #    related_project = projects_df.loc[prj['corpus_id']]
    #    scores.append(prj['score'])
    #    df = df.append(related_project) #deprecated but couldn't get pd.concat to work
    #df.insert(0, "cosim_score", scores)
    return df

try:
    df = search_projects(title, abstract, numResults)
    st.dataframe(df)
    with c2: # Map demo
        map_data = df[['latitude','longitude','Institution-Name']].dropna()
        st.subheader('Matching Research Institutions')
	
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

#st.button("Re-run")
