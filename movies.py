# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:10:13 2018

@author: Fahad Hilal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ts_dataset = pd.read_csv('testdata.tsv', delimiter = '\t', quoting = 3, header=None)
tr_dataset = pd.read_csv('training.tsv', delimiter = '\t', quoting = 3, header=None)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
tr_corpus = []
for i in range(0, 7086):
    review = re.sub('[^a-zA-Z]', ' ', tr_dataset[1][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    tr_corpus.append(review)

ts_corpus = []
for i in range(0, 33052):
    review = re.sub('[^a-zA-Z]', ' ', ts_dataset[0][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    ts_corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1707)
X = cv.fit_transform(tr_corpus).toarray()
y = tr_dataset.iloc[:, 0].values

## Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X, y)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1707)
X_test = cv.fit_transform(ts_corpus).toarray()


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)