import streamlit as st
import pandas as pd
import plotly.express as ex
import nltk.corpus
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import re
import imdb 
ia = imdb.IMDb()



nltk.download('stopwords')
stop = stopwords.words('english')
df=pd.read_csv('df_pro.csv')
tfidf_movieid=pickle.load(open('tfidf_movieid.pickle','rb'))
tsvd_mat=pickle.load(open('tsvd_mat.pickle','rb'))
tsvd=pickle.load(open('tsvd.pickle','rb'))
def preprocess(x):
    x=str(x).replace('\n','').lower()
    x=re.sub(r"([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", x)
    x= " ".join([word for word in x.split() if word not in (stop)])
    return x

col1, col2, col3 = st.columns([1,6,1])
col2.image('images/movie-theatre.jpg')
col2.title('Movie search engine')
st.markdown('A streamlit fun app for searching forgotten movies names from describing the story line')

input_text=st.text_area(label='Describe your movie below',value='')
col1, col2, col3 = st.columns([1,1,1])
if col2.button('search'):

    query=preprocess(input_text)
    query_mat = tsvd.transform(tfidf_movieid.transform([query]))
    dist = pairwise_distances(X=tsvd_mat , Y=query_mat, metric='cosine')
    if np.min(dist) == 1.0:
            st.subheader('No movie found')


    else :
          col1, col2, col3 = st.columns([1,6,1])
          col2.subheader('Movie : '+str(df['Moviename'][np.argmin(dist.flatten())]))
          # searching the name 
          search = ia.search_movie(df['Moviename'][np.argmin(dist.flatten())])
          series = ia.get_movie(search[0].movieID)
        # getting cover url of the series
          cover = series.data['cover url']
          col2.image(
            cover,width=400 # Manually Adjust the width of the image as per requirement
        )
  



