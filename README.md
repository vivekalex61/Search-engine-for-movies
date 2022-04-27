
# Movie search engine 

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

A fun app for predicting the movie name from describing the story.



## Introduction 

#### What is movie search engine?

Search engine is a fancy word for this project, basically the app will find a movie whose description matches to the descriptions stored in the database.
The data is consist of 11850 movies informations from IMDB.
## Overview of Movie search engine
- Datasets and Data-Loading
- Text Preprocessing
- Model creation

### Datasets and Data-Loading

Dataset was collected from www.imdb.com.It consist 11850 movies and columns consist of 'Moviename', 'rating', 'genre',
       'director', 'cast', 'writer' and 'summary'.
### Text Preprocessing

The aim of pre-processing is an improvement of the image data that suppresses undesired distortions or enhances some image features relevant for further processing and analysis task.
Text preprocessing is a method to clean the text data and make it ready to feed data to the model.
 Text data contains noise in various forms like emotions, punctuation, text in a different case.
When we talk about Human Language then, there are different ways to say the same thing, And this is only the main problem we have to deal with because machines will not understand words,
they need numbers so we need to convert text to numbers in an efficient manner.

Ref : https://www.analyticsvidhya.com/blog/2021/06/must-known-techniques-for-text-preprocessing-in-nlp/

It includes:
- Lowercase string
- removing stopwords
- Removing symbols and Punctuations
- Removing unwanted spaces between words


### Model building and training
Basic idea of the model is to predict the movie from description.
The model is converted into vectors and SVD is used to reduce the dimension of the vectors. 
Finally cosine similarity algorithm is used to deteremine the similiarity between query vector and the saved vectors.

#### TF-IDF 
TF-IDF (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.

This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.

TF-IDF for a word in a document is calculated by multiplying two different metrics:
1,term frequency of a word in a document, 2,inverse document frequency of the word across a set of documents.

TF-IDF algorithms help sort data into categories, as well as extract keywords.

#### TruncatedSVD 

This transformer performs linear dimensionality reduction by means of truncated singular value decomposition (SVD). Contrary to PCA, this estimator does not center the data before computing the singular value decomposition. This means it can work with sparse matrices efficiently.
converting to number of components = 200
 #### Cosine Similarity

Cosine similarity measures the similarity between two vectors of an inner product space. It is measured by the cosine of the angle between two vectors and determines whether two vectors are pointing in roughly the same direction. It is often used to measure document similarity in text analysis

Cosine similarity captures the orientation (the angle) of the documents and not the magnitude. The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance they could still have a smaller angle between them. Smaller the angle, higher the similarity.
![alt text](https://raw.githubusercontent.com/vivekalex61/insightsearch/master/test/overall_sentiments.png)
![alt text](https://raw.githubusercontent.com/vivekalex61/insightsearch/master/test/overall_sentiments.png)
![alt text](https://raw.githubusercontent.com/vivekalex61/insightsearch/master/test/overall_sentiments.png)

## Predictions
![alt text](https://raw.githubusercontent.com/vivekalex61/insightsearch/master/test/overall_sentiments.png)
![alt text](https://raw.githubusercontent.com/vivekalex61/insightsearch/master/test/overall_sentiments.png)


## End Notes

The application need minimum of 5 words to precdict the movie accuratly . Prediction can went wrong if a movie has several seasons or parts.
