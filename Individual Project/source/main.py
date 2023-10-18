import numpy as np
import pandas as pd
import os
import math
import pickle

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import GenericUnivariateSelect, RFECV, chi2
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from scipy.stats import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

current_directory = os.getcwd()
dataset = 'CDs_and_Vinyl'

review_training_df = pd.read_json(os.path.join(current_directory, dataset, 'train', 'review_training.json'))
product_training_df = pd.read_json(os.path.join(current_directory, dataset, 'train', 'product_training.json'))

review_test_df = pd.read_json(os.path.join(current_directory, dataset, 'test3', 'review_test.json'))
product_test_df = pd.read_json(os.path.join(current_directory, dataset, 'test3', 'product_test.json'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    stop_words = stopwords.words('english')
    stop_tokens = [token for token in tokens if token not in stop_words]
    lemmatized = [WordNetLemmatizer().lemmatize(token) for token in stop_tokens]
    processed = ' '.join(lemmatized)
    return processed

# applying to training data
review_training_df['reviewText'] = review_training_df['reviewText'].fillna('')
review_training_df['reviewText'] = review_training_df['reviewText'].apply(preprocess)

# applying to test data
review_test_df['reviewText'] = review_test_df['reviewText'].fillna('')
review_test_df['reviewText'] = review_test_df['reviewText'].apply(preprocess)

# feature selection for tfidf
reviewText_train = review_training_df.groupby('asin')['reviewText'].agg(lambda x: ' '.join(x)).reset_index()
reviewText_train.rename(columns={"reviewText":"joinedReviews"}, inplace=True)

reviewText_test = review_test_df.groupby('asin')['reviewText'].agg(lambda x: ' '.join(x)).reset_index()
reviewText_test.rename(columns={"reviewText":"joinedReviews"}, inplace=True)

training_df = pd.merge(product_training_df, reviewText_train, on='asin', how = 'left')
testing_df = pd.merge(product_test_df, reviewText_test, on='asin', how = 'left')

# model training
TRAIN_X = training_df['joinedReviews']
TRAIN_Y = training_df['awesomeness']
TEST = testing_df['joinedReviews']

tfidf = TfidfVectorizer()
tf_train = tfidf.fit_transform(TRAIN_X)
tf_test = tfidf.transform(TEST)

# logit
logit = LogisticRegression(max_iter=1000)
logit.fit(tf_train, TRAIN_Y)

# random forest
rf = RandomForestClassifier(n_estimators = 48, max_depth = 2, n_jobs=-1)
rf.fit(tf_train, TRAIN_Y)

# decision tree
tree = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
tree = tree.fit(tf_train, TRAIN_Y)

# late fusion
estimators=[('rf', rf), ('tree', tree), ('log', logit)]
ensemble = VotingClassifier(estimators, voting='soft')
ensemble.fit(tf_train, TRAIN_Y)

# saving model
pickle.dump(ensemble, open('model.pkl', 'wb'))

# results
lf_pred = ensemble.predict(tf_test)
product_test_df['prediction'] = lf_pred
with open('results.json', 'w') as f:
    f.write(product_test_df.to_json())