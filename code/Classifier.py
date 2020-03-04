from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

import numpy as np
import os

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFECV
#imports
import nltk #import the natural language toolkit library
from nltk.stem.snowball import FrenchStemmer #import the French stemming library
from nltk.corpus import stopwords #import stopwords from nltk corpus
import re #import the regular expressions library
from collections import Counter #allows for counting the number of occurences in a list


#env_topics = pd.read_csv('/users/marion1meyers/documents/lawtechlab/environmental-args-belgian-law/all_topics_juridict/saved_dataframes/topics_env_NEW.csv')
#non_env_topics = pd.read_csv('/users/marion1meyers/documents/lawtechlab/environmental-args-belgian-law/all_topics_juridict/saved_dataframes/topics_non_env_NEW.csv')

case_topic_termdoc = pd.read_csv('/users/marion1meyers/documents/lawtechlab/environmental-args-belgian-law/all_topics_juridict/saved_dataframes/case_topic_term_non_env_NEW.csv')
#texts=np.zeros(len(case_topic_termdoc))


small_df = case_topic_termdoc.loc[0:19]
texts = list()
for i in range(0,len(small_df)):
    texts.append('hello how are you doing in this wolrd, keeping up?')
"""for i, row in small_df.iterrows():
    print(i)
    row.text = ['hello','hola','hi']
    small_df.iloc[i]['text'] = ['hello','hola','hi']
    print(small_df.iloc[i]['text'])"""
small_df['text'] = texts
print(small_df.head())
target0 = np.zeros(10)
target1=np.ones(10)
targets = np.concatenate([target0,target1])
print(len(targets))
final_df = pd.DataFrame({'text':small_df['text'], 'label':targets})



dataframe = pd.read_csv()

count_vectorizer = CountVectorizer()
word_counts = count_vectorizer.fit_transform(final_df['text'].values)

classifier = LogisticRegression(class_weight='balanced')
classifier = RandomForestClassifier()
scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
scores = cross_validate(classifier, word_counts, final_df.label.values, scoring=scoring, cv=5)

sorted(scores.keys())
LR_fit_time = scores['fit_time'].mean()
LR_score_time = scores['score_time'].mean()
LR_accuracy = scores['test_accuracy'].mean()
LR_precision = scores['test_precision_macro'].mean()
LR_recall = scores['test_recall_macro'].mean()
LR_f1 = scores['test_f1_weighted'].mean()
LR_roc = scores['test_roc_auc'].mean()

LR_fit_time_sd = scores['fit_time'].std()
LR_score_time_sd = scores['score_time'].std()
LR_accuracy_sd = scores['test_accuracy'].std()
LR_precision_sd = scores['test_precision_macro'].std()
LR_recall_sd = scores['test_recall_macro'].std()
LR_f1_sd = scores['test_f1_weighted'].std()
LR_roc_sd = scores['test_roc_auc'].std()

print('accuracy : '+str(LR_accuracy))
print(LR_fit_time,LR_score_time,LR_accuracy,LR_precision,LR_recall,LR_f1,LR_roc)
print(LR_fit_time_sd,LR_score_time_sd,LR_accuracy_sd,LR_precision_sd,LR_recall_sd,LR_f1_sd,LR_roc_sd)

