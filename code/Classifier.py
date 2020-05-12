#!/usr/bin/env python
# coding: utf-8



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from nltk.corpus import stopwords

import spacy
nlp = spacy.load('fr_core_news_md')

from string import punctuation

# In[3]:


def run_classifier(classifier, input_features, labels, scoring):
    print(classifier)
    scores = cross_validate(classifier, input_features, labels, scoring=scoring, cv=10)
    y_pred = cross_val_predict(classifier, input_features,labels , cv=10)
    conf_mat = confusion_matrix(labels, y_pred)
    print('confusion matrix : '+str(conf_mat))

    sorted(scores.keys())
    fit_time = scores['fit_time'].mean()
    score_time = scores['score_time'].mean()
    accuracy = scores['test_accuracy'].mean()
    precision = scores['test_precision_macro'].mean()
    recall = scores['test_recall_macro'].mean()
    f1 = scores['test_f1_weighted'].mean()
    roc = scores['test_roc_auc'].mean()

    fit_time_sd = scores['fit_time'].std()
    score_time_sd = scores['score_time'].std()
    accuracy_sd = scores['test_accuracy'].std()
    precision_sd = scores['test_precision_macro'].std()
    recall_sd = scores['test_recall_macro'].std()
    f1_sd = scores['test_f1_weighted'].std()
    roc_sd = scores['test_roc_auc'].std()

    print('accuracy : ' + str(accuracy))
    print('accuracy std : '+str(accuracy_sd))
    print('precision : '+str(precision))
    print('precision sd: '+str(precision_sd))
    print('recall : '+str(recall))
    print('recall sd : '+str(recall_sd))
    print('f1 : '+str(f1))
    print('f1 sd : '+str(f1_sd))
    print('roc :'+str(roc))
    print('roc sd:'+str(roc_sd))
    return scores, conf_mat, y_pred





def get_stopswords(type="veronis"):
    #returns the veronis stopwords in unicode, or if any other value is passed, it returns the default nltk french stopwords
    if type=="veronis":
        #VERONIS STOPWORDS Jean Veronis is a French Linguist
        raw_stopword_list = ["Ap.", "Apr.", "GHz", "MHz", "USD", "a", "afin", "ah", "ai", "aie", "aient", "aies", "ait", "alors", "après", "as", "attendu", "au", "au-delà", "au-devant", "aucun", "aucune", "audit", "auprès", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autour", "autre", "autres", "autrui", "aux", "auxdites", "auxdits", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons", "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était", "car", "ce", "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "celà", "cent", "cents", "cependant", "certain", "certaine", "certaines", "certains", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "cf.", "cg", "cgr", "chacun", "chacune", "chaque", "chez", "ci", "cinq", "cinquante", "cinquante-cinq", "cinquante-deux", "cinquante-et-un", "cinquante-huit", "cinquante-neuf", "cinquante-quatre", "cinquante-sept", "cinquante-six", "cinquante-trois", "cl", "cm", "cm²", "comme", "contre", "d", "d'", "d'après", "d'un", "d'une", "dans", "de", "depuis", "derrière", "des", "desdites", "desdits", "desquelles", "desquels", "deux", "devant", "devers", "dg", "différentes", "différents", "divers", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dl", "dm", "donc", "dont", "douze", "du", "dudit", "duquel", "durant", "dès", "déjà", "e", "eh", "elle", "elles", "en", "en-dehors", "encore", "enfin", "entre", "envers", "es", "est", "et", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eûmes", "eût", "eûtes", "f", "fait", "fi", "flac", "fors", "furent", "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gr", "h", "ha", "han", "hein", "hem", "heu", "hg", "hl", "hm", "hm³", "holà", "hop", "hormis", "hors", "huit", "hum", "hé", "i", "ici", "il", "ils", "j", "j'", "j'ai", "j'avais", "j'étais", "jamais", "je", "jusqu'", "jusqu'au", "jusqu'aux", "jusqu'à", "jusque", "k", "kg", "km", "km²", "l", "l'", "l'autre", "l'on", "l'un", "l'une", "la", "laquelle", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "lez", "lors", "lorsqu'", "lorsque", "lui", "lès", "m", "m'", "ma", "maint", "mainte", "maintes", "maints", "mais", "malgré", "me", "mes", "mg", "mgr", "mil", "mille", "milliards", "millions", "ml", "mm", "mm²", "moi", "moins", "mon", "moyennant", "mt", "m²", "m³", "même", "mêmes", "n", "n'avait", "n'y", "ne", "neuf", "ni", "non", "nonante", "nonobstant", "nos", "notre", "nous", "nul", "nulle", "nº", "néanmoins", "o", "octante", "oh", "on", "ont", "onze", "or", "ou", "outre", "où", "p", "par", "par-delà", "parbleu", "parce", "parmi", "pas", "passé", "pendant", "personne", "peu", "plus", "plus_d'un", "plus_d'une", "plusieurs", "pour", "pourquoi", "pourtant", "pourvu", "près", "puisqu'", "puisque", "q", "qu", "qu'", "qu'elle", "qu'elles", "qu'il", "qu'ils", "qu'on", "quand", "quant", "quarante", "quarante-cinq", "quarante-deux", "quarante-et-un", "quarante-huit", "quarante-neuf", "quarante-quatre", "quarante-sept", "quarante-six", "quarante-trois", "quatorze", "quatre", "quatre-vingt", "quatre-vingt-cinq", "quatre-vingt-deux", "quatre-vingt-dix", "quatre-vingt-dix-huit", "quatre-vingt-dix-neuf", "quatre-vingt-dix-sept", "quatre-vingt-douze", "quatre-vingt-huit", "quatre-vingt-neuf", "quatre-vingt-onze", "quatre-vingt-quatorze", "quatre-vingt-quatre", "quatre-vingt-quinze", "quatre-vingt-seize", "quatre-vingt-sept", "quatre-vingt-six", "quatre-vingt-treize", "quatre-vingt-trois", "quatre-vingt-un", "quatre-vingt-une", "quatre-vingts", "que", "quel", "quelle", "quelles", "quelqu'", "quelqu'un", "quelqu'une", "quelque", "quelques", "quelques-unes", "quelques-uns", "quels", "qui", "quiconque", "quinze", "quoi", "quoiqu'", "quoique", "r", "revoici", "revoilà", "rien", "s", "s'", "sa", "sans", "sauf", "se", "seize", "selon", "sept", "septante", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "si", "sinon", "six", "soi", "soient", "sois", "soit", "soixante", "soixante-cinq", "soixante-deux", "soixante-dix", "soixante-dix-huit", "soixante-dix-neuf", "soixante-dix-sept", "soixante-douze", "soixante-et-onze", "soixante-et-un", "soixante-et-une", "soixante-huit", "soixante-neuf", "soixante-quatorze", "soixante-quatre", "soixante-quinze", "soixante-seize", "soixante-sept", "soixante-six", "soixante-treize", "soixante-trois", "sommes", "son", "sont", "sous", "soyez", "soyons", "suis", "suite", "sur", "sus", "t", "t'", "ta", "tacatac", "tandis", "te", "tel", "telle", "telles", "tels", "tes", "toi", "ton", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "trente-cinq", "trente-deux", "trente-et-un", "trente-huit", "trente-neuf", "trente-quatre", "trente-sept", "trente-six", "trente-trois", "trois", "très", "tu", "u", "un", "une", "unes", "uns", "v", "vers", "via", "vingt", "vingt-cinq", "vingt-deux", "vingt-huit", "vingt-neuf", "vingt-quatre", "vingt-sept", "vingt-six", "vingt-trois", "vis-à-vis", "voici", "voilà", "vos", "votre", "vous", "w", "x", "y", "z", "zéro", "à", "ç'", "ça", "ès", "étaient", "étais", "était", "étant", "étiez", "étions", "été", "étée", "étées", "étés", "êtes", "être", "ô"]
    else:
        #get French stopwords from the nltk kit
        raw_stopword_list = stopwords.words('french') #create a list of all French stopwords
    #stopword_list = [word.decode('utf8') for word in raw_stopword_list] #make to decode the French stopwords as unicode objects rather than ascii
    
    #also add the punctuation signs to the list of stopwords
    signs = list(set(punctuation))
    stopwords_law=["xiii", "xiiir", "viii","xi", "vi", "viiir", "xiiie", "xxx"]

    raw_stopword_list = raw_stopword_list + signs +stopwords_law
    return raw_stopword_list



# ## Labelling correction:
# From the error analysis performed after the data set was classified, we realized some errors in the labelling rules.
# Namely, some subjects under 'Varia' were classified as environmental because they had the word 'nature' in them
# (ie. Varia/Calamités naturelles/). It is too computationally intensive to re-run the whole thing with adding this rule
# so we will simply change their label directly in all the data tables.

# ### Topic Tables Correction



"""for column in topics_env_NEW.columns:
    if 'Varia' in column:
        #copy the column to the non_env topic table
        print('non env topic shape : '+str(topics_non_env_NEW.shape))
        topics_non_env_NEW[column] = topics_env_NEW[column]
        print('non env topic shape after adding: '+str(topics_non_env_NEW.shape))
        print('env topic shape : '+str(topics_env_NEW.shape))
        del topics_env_NEW[column]
        print('env topic shape after deleting : '+str(topics_env_NEW.shape))

topics_non_env_NEW.to_csv('/users/marion1meyers/documents/lawtechlab/environmental-args-belgian-law/all_topics_juridict/all_topics_csv/topics_non_env_NEW.csv')
topics_env_NEW.to_csv('/users/marion1meyers/documents/lawtechlab/environmental-args-belgian-law/all_topics_juridict/all_topics_csv/topics_env_NEW.csv')
"""


# ### Case topic correction


# This was done both for the clean_vocab and non clean_vocab case tables
def correct_varia_topic(case_topic_termdoc):
    #This was done both for the clean_vocab and non clean_vocab case tables
    """case_topic_termdoc_non_env= pd.read_csv('/users/marion1meyers/documents/lawtechlab/environmental-args-belgian-law/all_topics_juridict/saved_dataframes/case_topic_term_non_env_full_NEW_clean_vocab.csv')
    #case_topic_termdoc_env = pd.read_csv('/users/marion1meyers/documents/lawtechlab/environmental-args-belgian-law/all_topics_juridict/saved_dataframes/case_topic_term_env_full_NEW_clean_vocab.csv')
    case_topic_termdoc_non_env =case_topic_termdoc_non_env.drop(len(case_topic_termdoc_non_env)-1)

    for index, case in case_topic_termdoc_env.iterrows():
        if 'Varia' in case['topic']:
            print(case['topic'])
            print('non env shape before '+str(case_topic_termdoc_non_env.shape))
            case_topic_termdoc_non_env = case_topic_termdoc_non_env.append(case, ignore_index= True)
            print('non env shape after '+str(case_topic_termdoc_non_env.shape))

            print('env shape before : '+str(case_topic_termdoc_env.shape))
            #case_topic_termdoc_env = case_topic_termdoc_env.drop(case_topic_termdoc_env.index[index-1])
            case_topic_termdoc_env.drop(index, inplace=True)
            print('env shape after : '+str(case_topic_termdoc_env.shape))

    #double-check if the transfer has worked
    for index, case in case_topic_termdoc_non_env.iterrows():
        if 'Varia' in case['topic']:
            print(case)

    #save the new files
    case_topic_termdoc_non_env.to_csv('/users/marion1meyers/documents/lawtechlab/environmental-args-belgian-law/all_topics_juridict/saved_dataframes/case_topic_term_non_env_full_NEW_clean_vocab.csv')
    #case_topic_termdoc_env.to_csv('/users/marion1meyers/documents/lawtechlab/environmental-args-belgian-law/all_topics_juridict/saved_dataframes/case_topic_term_env_full_NEW_clean_vocab.csv')
    """


    # In[10]:

    """counter=0
    for index, case in case_topic_termdoc_env.iterrows():
        if 'Varia' in case['topic']:
            print(case.topic)
            counter = counter +1
    
    counter2=0
    for index, case in case_topic_termdoc_non_env.iterrows():
        if 'Varia' in case['topic']:
            print(case.topic)
            counter2 = counter2 +1
    print(counter)
    print(counter2)"""



def read_and_preprocess_datasets(datasets_path):

    case_topic_termdoc_non_env = pd.read_csv(datasets_path+"/case_topic_non_env.csv")
    case_topic_termdoc_non_env_lemmatized = pd.read_csv(datasets_path+"/case_topic_non_env_lemmatized.csv")
    case_topic_termdoc_non_env_no_stopwords=pd.read_csv(datasets_path+"/case_topic_non_env_no_stopwords.csv")


    case_topic_termdoc_env = pd.read_csv(datasets_path+"case_topic_env.csv")
    case_topic_termdoc_env_lemmatized = pd.read_csv(datasets_path+"/case_topic_env_lemmatized.csv")
    case_topic_termdoc_env_no_stopwords = pd.read_csv(datasets_path+"/case_topic_env_no_stopwords.csv")


    print("INITIAL SHAPES")
    print('non env shape:' +str(case_topic_termdoc_non_env.shape))
    print('env shape: '+str(case_topic_termdoc_env.shape))

    print('non env lemmatized shape:' +str(case_topic_termdoc_non_env_lemmatized.shape))
    print('env lemmatized shape: '+str(case_topic_termdoc_env_lemmatized.shape))

    print('non env no stopwords shape:' +str(case_topic_termdoc_non_env_no_stopwords.shape))
    print('env no stopwords shape: '+str(case_topic_termdoc_env_no_stopwords.shape))

    #delete possible duplicates
    case_topic_termdoc_non_env = case_topic_termdoc_non_env.drop_duplicates(subset="case_number", keep='first')
    case_topic_termdoc_env =case_topic_termdoc_env.drop_duplicates(subset="case_number", keep='first')

    case_topic_termdoc_non_env_lemmatized = case_topic_termdoc_non_env_lemmatized.drop_duplicates(subset="case_number", keep='first')
    case_topic_termdoc_env_lemmatized =case_topic_termdoc_env_lemmatized.drop_duplicates(subset="case_number", keep='first')

    case_topic_termdoc_non_env_no_stopwords = case_topic_termdoc_non_env_no_stopwords.drop_duplicates(subset="case_number", keep='first')
    case_topic_termdoc_env_no_stopwords =case_topic_termdoc_env_no_stopwords.drop_duplicates(subset="case_number", keep='first')

    print("SHAPES WITH NO DUPLICATES")
    print('non env shape:' +str(case_topic_termdoc_non_env.shape))
    print('env shape: '+str(case_topic_termdoc_env.shape))

    print('non env lemmatized shape:' +str(case_topic_termdoc_non_env_lemmatized.shape))
    print('env lemmatized shape: '+str(case_topic_termdoc_env_lemmatized.shape))

    print('non env no stopwords shape:' +str(case_topic_termdoc_non_env_no_stopwords.shape))
    print('env no stopwords shape: '+str(case_topic_termdoc_env_no_stopwords.shape))

    filled_non_env = case_topic_termdoc_non_env.fillna('')
    filled_env =case_topic_termdoc_env.fillna('')

    filled_non_env_lemmatized = case_topic_termdoc_non_env_lemmatized.fillna('')
    filled_env_lemmatized =case_topic_termdoc_env_lemmatized.fillna('')

    filled_non_env_no_stopwords = case_topic_termdoc_non_env_no_stopwords.fillna('')
    filled_env_no_stopwords =case_topic_termdoc_env_no_stopwords.fillna('')


    #only keep the rows that have some text.
    filled_non_env = filled_non_env[filled_non_env.case_text != '']
    filled_env = filled_env[filled_env.case_text != '']

    filled_non_env_lemmatized = filled_non_env_lemmatized[filled_non_env_lemmatized.case_text != '']
    filled_env_lemmatized = filled_env_lemmatized[filled_env_lemmatized.case_text != '']

    filled_non_env_no_stopwords= filled_non_env_no_stopwords[filled_non_env_no_stopwords.case_text != '']
    filled_env_no_stopwords = filled_env_no_stopwords[filled_env_no_stopwords.case_text != '']

    print("SHAPES WIHTOUT EMPTY TEXTS")
    print('non env filled shape: '+str(filled_non_env.shape))
    print('env filled shape: '+str(filled_env.shape))

    print('non env lemm filled shape: '+str(filled_non_env_lemmatized.shape))
    print('env lemm filled shape: '+str(filled_env_lemmatized.shape))

    print('non env no stopwords filled shape: '+str(filled_non_env_no_stopwords.shape))
    print('env no stopwords filled shape: '+str(filled_env_no_stopwords.shape))




    #ENVIRONMENTAl
    #unique = case_numbers_lemm.where(case_numbers_lemm not in case_numbers_no_stop)
    df = pd.concat([filled_env_lemmatized, filled_env_no_stopwords]) # concat dataframes
    df = df.reset_index(drop=True) # reset the index
    df_gpby = df.groupby(list(df['case_number'])) #group by
    idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1] #reindex


    env_to_del = list()
    for el in idx:
        #print(type(non_env_to_del))
        #print(type(df.loc[el].case_number))
        env_to_del.append(df.loc[el].case_number)
        #print(non_env_to_del)
        #print(df.loc[el].term_doc_matrix)
    print(len(env_to_del))

    filled_env = filled_env[~filled_env['case_number'].isin(env_to_del)]
    filled_env_lemmatized = filled_env_lemmatized[~filled_env_lemmatized['case_number'].isin(env_to_del)]


    #NON-ENVIRONMENTAL
    #unique = case_numbers_lemm.where(case_numbers_lemm not in case_numbers_no_stop)
    df = pd.concat([filled_non_env_lemmatized, filled_non_env_no_stopwords]) # concat dataframes
    df = df.reset_index(drop=True) # reset the index
    df_gpby = df.groupby(list(df['case_number'])) #group by
    idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1] #reindex



    non_env_to_del = list()
    for el in idx:
        #print(type(non_env_to_del))
        #print(type(df.loc[el].case_number))
        non_env_to_del.append(df.loc[el].case_number)
        #print(non_env_to_del)
        #print(df.loc[el].term_doc_matrix)
    print(len(non_env_to_del))


    # In[15]:


    filled_non_env = filled_non_env[~filled_non_env['case_number'].isin(non_env_to_del)]
    filled_non_env_lemmatized = filled_non_env_lemmatized[~filled_non_env_lemmatized['case_number'].isin(non_env_to_del)]



    print("SHAPES WIHTOUT EMPTY TEXTS")
    print('non env filled shape: '+str(filled_non_env.shape))
    print('env filled shape: '+str(filled_env.shape))

    print('non env lemm filled shape: '+str(filled_non_env_lemmatized.shape))
    print('env lemm filled shape: '+str(filled_env_lemmatized.shape))

    print('non env no stopwords filled shape: '+str(filled_non_env_no_stopwords.shape))
    print('env no stopwords filled shape: '+str(filled_env_no_stopwords.shape))

    return filled_env, filled_non_env, filled_env_lemmatized, filled_non_env_lemmatized, filled_env_no_stopwords, filled_non_env_no_stopwords



def label_datasets(filled_env, filled_non_env, filled_env_lemmatized,filled_non_env_lemmatized, filled_env_no_stopwords , filled_non_env_no_stopwords):
    """
    NON-ENV: LABELLED AS ZEROS
    ENV : LABELLED AS ONES"""

    small_non_env = pd.DataFrame({'case_number':filled_non_env['case_number'], 'topic':filled_non_env['topic'], 'case_text':filled_non_env['case_text'], 'label':np.zeros(len(filled_non_env))})
    small_env = pd.DataFrame({'case_number':filled_env['case_number'], 'topic':filled_env['topic'],'case_text':filled_env['case_text'], 'label':np.ones(len(filled_env))})
    case_dataset = small_non_env.append(small_env, ignore_index=True)

    small_non_env_lemm = pd.DataFrame({'case_number':filled_non_env_lemmatized['case_number'], 'topic':filled_non_env_lemmatized['topic'], 'case_text':filled_non_env_lemmatized['case_text'], 'label':np.zeros(len(filled_non_env_lemmatized))})
    small_env_lemm = pd.DataFrame({'case_number':filled_env_lemmatized['case_number'], 'topic':filled_env_lemmatized['topic'],'case_text':filled_env_lemmatized['case_text'], 'label':np.ones(len(filled_env_lemmatized))})
    case_dataset_lemm = small_non_env_lemm.append(small_env_lemm, ignore_index=True)


    small_non_env_ns= pd.DataFrame({'case_number':filled_non_env_no_stopwords['case_number'], 'topic':filled_non_env_no_stopwords['topic'], 'case_text':filled_non_env_no_stopwords['case_text'], 'label':np.zeros(len(filled_non_env_no_stopwords))})
    small_env_ns = pd.DataFrame({'case_number':filled_env_no_stopwords['case_number'], 'topic':filled_env_no_stopwords['topic'],'case_text':filled_env_no_stopwords['case_text'], 'label':np.ones(len(filled_env_no_stopwords))})
    case_dataset_ns = small_non_env_ns.append(small_env_ns, ignore_index=True)

    print(case_dataset.shape)
    print(case_dataset_lemm.shape)
    print(case_dataset_ns.shape)

    return case_dataset, case_dataset_lemm, case_dataset_ns


#GET THE DATASETS
datasets_path=""
case_dataset, case_dataset_lemm, case_dataset_ns = label_datasets(read_and_preprocess_datasets(datasets_path))

# NON-LEMMATIZED DATASET (not useful actually..)
"""count_vectorizer = CountVectorizer(stop_words=get_stopswords())
td_idf_vectorizer = TfidfVectorizer(min_df=1, stop_words=get_stopswords())
word_counts = count_vectorizer.fit_transform(case_dataset['case_text'].values)
td_idfs = td_idf_vectorizer.fit_transform(case_dataset['case_text'].values)"""



"""classifier1 = LogisticRegression(class_weight='balanced')
classifier2 = RandomForestClassifier(class_weight='balanced')
classifier3 = SVC(class_weight='balanced')
classifier4 = MultinomialNB()
classifier5=linear_model.SGDClassifier(loss='perceptron',class_weight='balanced')
classifier6 = DecisionTreeClassifier(class_weight='balanced')
scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

classifiers = [classifier1, classifier2, classifier3]#, classifier2, classifier3, classifier4, classifier5]
labels = case_dataset.label.values

#print('RUN WITH BAG OF WORDS (WORD COUNTS) METHOD')
#for classifier in classifiers:
#wc_scores, wc_conf_mat, wc_y_pred = run_classifier(classifier1, word_counts,labels, scoring)

#print('RUN WITH TD-IDF METHOD')
#for classifier in classifiers:
tdidf_scores, tdidf_conf_mat, tdidf_y_pred = run_classifier(classifier1, td_idfs, labels, scoring)"""


# FULLY LEMMATIZED DATA SET

count_vectorizer_lemm = CountVectorizer(stop_words=get_stopswords())
td_idf_vectorizer_lemm = TfidfVectorizer(min_df=1, stop_words=get_stopswords())
word_counts_lemm = count_vectorizer_lemm.fit_transform(case_dataset_lemm['case_text'].values)
td_idfs_lemm = td_idf_vectorizer_lemm.fit_transform(case_dataset_lemm['case_text'].values)


classifier1 = LogisticRegression(class_weight='balanced', max_iter=3000)
classifier2 = RandomForestClassifier(class_weight='balanced')
classifier3 = SVC(class_weight='balanced')
classifier4 = MultinomialNB()
classifier5=linear_model.SGDClassifier(class_weight='balanced')
classifier6 = DecisionTreeClassifier(class_weight='balanced')

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

classifiers = [classifier1, classifier2, classifier3]#, classifier2, classifier3, classifier4, classifier5]
labels = case_dataset_lemm.label.values

#print('RUN WITH BAG OF WORDS (WORD COUNTS) METHOD')
#for classifier in classifiers:
#wc_scores, wc_conf_mat, wc_y_pred = run_classifier(classifier1, word_counts_lemm,labels, scoring)

#print('RUN WITH TD-IDF METHOD')
#for classifier in classifiers:
#wc_scores, wc_conf_mat, wc_y_pred = run_classifier(classifier5, word_counts_lemm, labels, scoring)
tdidf_scores, tdidf_conf_mat, tdidf_y_pred = run_classifier(classifier4, td_idfs_lemm, labels, scoring)


# NON-LEMMATIZED DATASET, WITH STOPQWORDS AND NUMBERS TAKEN OUT

count_vectorizer_ns = CountVectorizer(stop_words=get_stopswords())
td_idf_vectorizer_ns = TfidfVectorizer(min_df=1, stop_words=get_stopswords())
word_counts_ns = count_vectorizer_ns.fit_transform(case_dataset_ns['case_text'].values)
td_idfs_ns = td_idf_vectorizer_ns.fit_transform(case_dataset_ns['case_text'].values)


classifier1 = LogisticRegression(class_weight='balanced')
classifier2 = RandomForestClassifier(class_weight='balanced')
classifier3 = SVC(class_weight='balanced')
classifier4 = MultinomialNB()
classifier5=linear_model.SGDClassifier(class_weight='balanced')
classifier6 = DecisionTreeClassifier(class_weight='balanced')

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

classifiers = [classifier1, classifier2, classifier3]#, classifier2, classifier3, classifier4, classifier5]
labels = case_dataset_ns.label.values

#print('RUN WITH BAG OF WORDS (WORD COUNTS) METHOD')
#for classifier in classifiers:
#wc_scores, wc_conf_mat, wc_y_pred = run_classifier(classifier1, word_counts_lemm,labels, scoring)

#print('RUN WITH TD-IDF METHOD')
#for classifier in classifiers:
wc_scores, wc_conf_mat, wc_y_pred = run_classifier(classifier5, word_counts_ns, labels, scoring)
#tdidf_scores, tdidf_conf_mat, tdidf_y_pred = run_classifier(classifier3, td_idfs_ns, labels, scoring)




#For a given vectorizer (either word count or td-idf based), a dataset (has to be in accordance with the vectorizer ofc), and a model
#check which words drive the classification
#and save the 100 biggest and 100 smallest to 2 csv files
def check_words_classifiers(vectorizer, dataset, model):
    vocabulary = vectorizer.get_feature_names()
    #model = linear_model.LogisticRegression(class_weight='balanced')
    labels = dataset.label.values

    model.fit(vectorizer, labels)

    coef = model.coef_[0]
    coef = pd.Series(coef)
    print(coef)

    #create small matrix with feature/coef
    coef_matrix = pd.DataFrame(columns=vocabulary)

    coef_matrix.loc[0] = np.exp(coef.values)
    #coef_matrix.loc[0] = coef.values

    coef_matrix.loc[0]

    coef_matrix = coef_matrix.sort_values(axis=1, by=0)



    print(coef_matrix.shape[1])
    #largest = coef_matrix.columns[coef_matrix.shape[1]-100:coef_matrix.shape[1]]
    #largest - coef_matrix.loc[coef_matrix.shape[1]-100:coef_matrix.shape[1]]
    largest= coef_matrix.iloc[:,coef_matrix.shape[1]-100:coef_matrix.shape[1]]
    smallest = coef_matrix.iloc[:,0:100]
    #df.loc[:,'Test_1':'Test_3']
    #largest_coef=coef_matrix.loc[0][coef_matrix.shape[1]-100:coef_matrix.shape[1]]
    #smallest = coef_matrix.loc[0:100]
    #coef_matrix.head()


    #reverse order in the larget so that the largest number actually goes firt
    largest = largest.sort_values(axis=1, by=0, ascending=False)
    for col in largest.columns:
        print(col)

    for small in smallest.columns:
        print(small)

    smallest.to_csv("non_env_coef_td_idfs.csv", index=False)
    largest.to_csv("env_coef_td_idfs.csv", index=False)

classifier = LogisticRegression(class_weight='balanced')

check_words_classifiers(td_idf_vectorizer_lemm, case_dataset_lemm, classifier)








