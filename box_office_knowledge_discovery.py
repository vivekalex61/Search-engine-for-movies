



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import necessary libraries

import os
import glob
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import re
import nltk.corpus


from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

nltk.download('stopwords')
stop = stopwords.words('english')
mergeddata=pd.read_csv('merged_data.csv')

df=mergeddata.drop('Unnamed: 0',axis=1)

df['all']=df['genre']+df['director']+df['cast']+df['writer']+df['summary']
df['all']=df['all'].apply(lambda x : str(x).replace('\n',''))
df['all']=df['all'].apply(lambda x : str(x).lower())
df['all']=df['all'].apply(lambda x : re.sub(r"([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", x))
#df['all']=df['all'].apply(lambda x : " ".join([word for word in x.split()]))
df['all']=df['all'].apply(lambda x : " ".join([word for word in x.split() if word not in (stop)]))
df.to_csv('df_pro.csv')

# Vectorizing pre-processed movie plots using TF-IDF
tfidfvec = TfidfVectorizer()
tfidfvec.fit(df['all'])
transformed=tfidfvec.transform(df['all'])
transformed=transformed.toarray()
pickle.dump(tfidfvec, open('tfidf_movieid.pickle','wb'))

df_t = pd.DataFrame(transformed,columns=tfidfvec.get_feature_names())
tsvd = TruncatedSVD(n_components=200)
tsvd.fit(df_t)
tsvd_mat=tsvd.transform(df_t)
pickle.dump(tsvd_mat, open('tsvd_mat.pickle','wb'))

pickle.dump(tsvd, open('tsvd.pickle','wb'))


