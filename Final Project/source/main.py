# CS 349 Final Project
# Ian Shi, Ellen Liao, Tim Fu

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

current_directory = os.getcwd()
dataset = 'CDs_and_Vinyl'

review_training_df = pd.read_json(os.path.join(current_directory, dataset, 'train', 'review_training.json'))
product_training_df = pd.read_json(os.path.join(current_directory, dataset, 'train', 'product_training.json'))
review_test_df = pd.read_json(os.path.join(current_directory, dataset, 'test2', 'review_test.json'))
product_test_df = pd.read_json(os.path.join(current_directory, dataset, 'test2', 'product_test.json'))

# sentiment analysis preprocessing
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    stop_words = stopwords.words('english')
    stop_tokens = [token for token in tokens if token not in stop_words]
    lemmatized = [WordNetLemmatizer().lemmatize(token) for token in stop_tokens]
    processed = ' '.join(lemmatized)
    return processed

analyzer = SentimentIntensityAnalyzer()

def comp_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = scores['compound']
    return sentiment

def sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] > 0:
        sentiment = 1
    else:
        sentiment = 0
    return sentiment

def pos_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = scores['pos']
    return sentiment

def neg_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = scores['neg']
    return sentiment

# applying to training data
review_training_df['reviewText'] = review_training_df['reviewText'].fillna('')
review_training_df['reviewText'] = review_training_df['reviewText'].apply(preprocess)
review_training_df['summary'] = review_training_df['summary'].fillna('')
review_training_df['summary'] = review_training_df['summary'].apply(preprocess)
# compound sentiment values
review_training_df['rev_compSentiment'] = review_training_df['reviewText'].apply(comp_sentiment)
review_training_df['summ_compSentiment'] = review_training_df['summary'].apply(comp_sentiment)
# 1 or 0 sentiment values
review_training_df['rev_Sentiment'] = review_training_df['reviewText'].apply(sentiment)
review_training_df['summ_Sentiment'] = review_training_df['summary'].apply(sentiment)
# pos sentiment values
review_training_df['rev_posSentiment'] = review_training_df['reviewText'].apply(pos_sentiment)
review_training_df['summ_posSentiment'] = review_training_df['summary'].apply(pos_sentiment)
# neg sentiment values
review_training_df['rev_negSentiment'] = review_training_df['reviewText'].apply(neg_sentiment)
review_training_df['summ_negSentiment'] = review_training_df['summary'].apply(neg_sentiment)

# applying to test data
review_test_df['reviewText'] = review_test_df['reviewText'].fillna('')
review_test_df['reviewText'] = review_test_df['reviewText'].apply(preprocess)
review_test_df['summary'] = review_test_df['summary'].fillna('')
review_test_df['summary'] = review_test_df['summary'].apply(preprocess)
# compound sentiment values
review_test_df['rev_compSentiment'] = review_test_df['reviewText'].apply(comp_sentiment)
review_test_df['summ_compSentiment'] = review_test_df['summary'].apply(comp_sentiment)
# 1 or 0 sentiment values
review_test_df['rev_Sentiment'] = review_test_df['reviewText'].apply(sentiment)
review_test_df['summ_Sentiment'] = review_test_df['summary'].apply(sentiment)
# pos sentiment values
review_test_df['rev_posSentiment'] = review_test_df['reviewText'].apply(pos_sentiment)
review_test_df['summ_posSentiment'] = review_test_df['summary'].apply(pos_sentiment)
# neg sentiment values
review_test_df['rev_negSentiment'] = review_test_df['reviewText'].apply(neg_sentiment)
review_test_df['summ_negSentiment'] = review_test_df['summary'].apply(neg_sentiment)

# preprocessing datasets for RFECV calculated features
train_reviews = review_training_df.groupby('asin')['reviewerID'].nunique()
test_reviews = review_test_df.groupby('asin')['reviewerID'].nunique()

train_ver = review_training_df.groupby('asin')['verified'].mean()
test_ver = review_test_df.groupby('asin')['verified'].mean()

train_rev_posSentiment = review_training_df.groupby('asin')['rev_posSentiment'].mean()
test_rev_posSentiment = review_test_df.groupby('asin')['rev_posSentiment'].mean()

train_rev_negSentiment = review_training_df.groupby('asin')['rev_negSentiment'].mean()
test_rev_negSentiment = review_test_df.groupby('asin')['rev_negSentiment'].mean()

train_summ_posSentiment = review_training_df.groupby('asin')['summ_posSentiment'].mean()
test_summ_posSentiment = review_test_df.groupby('asin')['summ_posSentiment'].mean()

train_summ_negSentiment = review_training_df.groupby('asin')['summ_negSentiment'].mean()
test_summ_negSentiment = review_test_df.groupby('asin')['summ_negSentiment'].mean()

train_rev_compSentiment = review_training_df.groupby('asin')['rev_compSentiment'].mean()
test_rev_compSentiment = review_test_df.groupby('asin')['rev_compSentiment'].mean()

train_summ_compSentiment = review_training_df.groupby('asin')['summ_compSentiment'].mean()
test_summ_compSentiment = review_test_df.groupby('asin')['summ_compSentiment'].mean()

training_df = pd.concat([train_reviews, train_ver, train_rev_posSentiment, train_rev_negSentiment, train_summ_posSentiment,
                         train_summ_negSentiment, train_rev_compSentiment, train_summ_compSentiment], axis = 1)
training_df = training_df.merge(product_training_df, left_index = True, right_on = 'asin')

testing_df = pd.concat([test_reviews, test_ver, test_rev_posSentiment, test_rev_negSentiment, test_summ_posSentiment,
                         test_summ_negSentiment, test_rev_compSentiment, test_summ_compSentiment], axis = 1)
testing_df = testing_df.merge(product_test_df, left_index = True, right_on = 'asin')

training_df['rev_posNegRatio'] = (training_df['rev_posSentiment'] + 1) \
    / (training_df['rev_negSentiment'] + 1)
training_df['summ_posNegRatio'] = (training_df['summ_posSentiment'] + 1) \
    / (training_df['summ_negSentiment'] + 1)
training_df['summToRev'] = (training_df['summ_compSentiment'] + 1) \
    / (training_df['rev_compSentiment'] + 1)

testing_df['rev_posNegRatio'] = (testing_df['rev_posSentiment'] + 1) \
    / (testing_df['rev_negSentiment'] + 1)
testing_df['summ_posNegRatio'] = (testing_df['summ_posSentiment'] + 1) \
    / (testing_df['summ_negSentiment'] + 1)
testing_df['summToRev'] = (testing_df['summ_compSentiment'] + 1) \
    / (testing_df['rev_compSentiment'] + 1)

# TRAINING DATA

review_training_df['vote'] = pd.to_numeric(review_training_df['vote'], errors='coerce')
review_training_df['vote'] = review_training_df['vote'].fillna(0).astype(int)
# max upvote
max_upvote = review_training_df.groupby('asin')['vote'].max()
max_upvote = max_upvote.values.reshape(-1, 1)

# normalized earliest rev from 0 to 10
earliest_rev = review_training_df.groupby('asin')['unixReviewTime'].min()
scaler = MinMaxScaler(feature_range=(0, 10))
normalized_rev = earliest_rev.values.reshape(-1, 1)
normalized_rev = scaler.fit_transform(normalized_rev)

# length of review
review_training_df['rev_word_count'] = review_training_df['reviewText'].str.split().apply(len)
av_word_count = review_training_df.groupby('asin')['rev_word_count'].mean()
av_word_count = av_word_count.values.reshape(-1,1)

# TOTAL number of reviews (we have unique reviews)
total_reviews = review_training_df.groupby('asin')['reviewerID'].count()
total_reviews = total_reviews.values.reshape(-1,1)

# difference in review time (normalized from 0 to 10)
time_diff = review_training_df.groupby('asin')['unixReviewTime'].apply(lambda x: x.max() - x.min())
norm_time_diff = time_diff.values.reshape(-1,1)
norm_time_diff = scaler.fit_transform(norm_time_diff)

training_df['normalized_time'] = normalized_rev
training_df['max_upvote'] = max_upvote
training_df['av_word_count'] = av_word_count
training_df['total_reviews'] = total_reviews
training_df['norm_time_diff'] = norm_time_diff

# TESTING DATA

review_test_df['vote'] = pd.to_numeric(review_test_df['vote'], errors='coerce')
review_test_df['vote'] = review_test_df['vote'].fillna(0).astype(int)
# max upvote
max_upvote = review_test_df.groupby('asin')['vote'].max()
max_upvote = max_upvote.values.reshape(-1, 1)

# normalized earliest rev from 0 to 10
earliest_rev = review_test_df.groupby('asin')['unixReviewTime'].min()
scaler = MinMaxScaler(feature_range=(0, 10))
normalized_rev = earliest_rev.values.reshape(-1, 1)
normalized_rev = scaler.fit_transform(normalized_rev)

# length of review
review_test_df['rev_word_count'] = review_test_df['reviewText'].str.split().apply(len)
av_word_count = review_test_df.groupby('asin')['rev_word_count'].mean()
av_word_count = av_word_count.values.reshape(-1,1)

# TOTAL number of reviews (we have unique reviews)
total_reviews = review_test_df.groupby('asin')['reviewerID'].count()
total_reviews = total_reviews.values.reshape(-1,1)

# difference in review time (normalized from 0 to 10)
time_diff = review_test_df.groupby('asin')['unixReviewTime'].apply(lambda x: x.max() - x.min())
norm_time_diff = time_diff.values.reshape(-1,1)
norm_time_diff = scaler.fit_transform(norm_time_diff)

testing_df['normalized_time'] = normalized_rev
testing_df['max_upvote'] = max_upvote
testing_df['av_word_count'] = av_word_count
testing_df['total_reviews'] = total_reviews
testing_df['norm_time_diff'] = norm_time_diff

features = training_df.groupby('awesomeness').mean(numeric_only = True)
features = features.columns

TRAIN_X = training_df[features]
TRAIN_Y = training_df['awesomeness']
TEST = testing_df[features]

rf = RandomForestClassifier(n_estimators = 28, max_depth = 1, n_jobs=-1)
rf = rf.fit(TRAIN_X, TRAIN_Y)

tree_features = ['verified', 'rev_negSentiment', 'summ_negSentiment', 'total_reviews', 'norm_time_diff']
tree = DecisionTreeClassifier(criterion='entropy', max_depth = 5)
tree = tree.fit(TRAIN_X[tree_features], TRAIN_Y)

log_reg = LogisticRegression(fit_intercept=False, solver='sag', multi_class='multinomial', n_jobs=-1)
lg_features = ['reviewerID', 'verified', 'summ_compSentiment', 'rev_posSentiment', 'summ_posSentiment', 'rev_negSentiment', 'summ_negSentiment', 'rev_posNegRatio', 'summ_posNegRatio', 'summToRev', 'normalized_time', 'max_upvote', 'av_word_count', 'total_reviews', 'norm_time_diff']
log_reg = log_reg.fit(TRAIN_X[lg_features], TRAIN_Y)

# late fusion
#create a dictionary of our models
estimators=[('rf', rf), ('tree', tree), ('log_reg', log_reg)]
#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='soft')
ensemble = ensemble.fit(TRAIN_X, TRAIN_Y)

pickle.dump(ensemble, open('model.pkl', 'wb'))

lf_pred = ensemble.predict(TEST)

product_test_df['prediction'] = lf_pred

with open('predictions.json', 'w') as f:
    f.write(product_test_df.to_json())