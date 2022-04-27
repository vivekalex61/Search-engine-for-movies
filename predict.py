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
print(df.columns)
tfidf_movieid=pickle.load(open('tfidf_movieid.pickle','rb'))
tsvd_mat=pickle.load(open('tsvd_mat.pickle','rb'))
tsvd=pickle.load(open('tsvd.pickle','rb'))
def preprocess(x):
    x=str(x).replace('\n','').lower()
    x=re.sub(r"([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", x)
    x= " ".join([word for word in x.split() if word not in (stop)])
    return x

q = "comedy series where characters are Sheldon and  Leonard."
q=preprocess(q)
query_mat = tsvd.transform(tfidf_movieid.transform([q]))
di= ['cosine']
for i in di:

        dist = pairwise_distances(X=tsvd_mat , Y=query_mat, metric=i)

        if np.min(dist) == 1.0:
            
            print('movie name unknown')
        else :
           
             
            print(df['Moviename'][np.argmin(dist.flatten())])
            

   
# searching the name 
search = ia.search_movie(df['Moviename'][np.argmin(dist.flatten())])
series = ia.get_movie(search[0].movieID)
  # getting cover url of the series
cover = series.data['cover url']
  
print(cover)







