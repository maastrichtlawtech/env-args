#!/usr/bin/env python
# coding: utf-8

import os
import re  # import the regular expressions library
from io import StringIO

import pdftotext

import nltk  # import the natural language toolkit library
import numpy as np
import pandas as pd
#from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from nltk.corpus import stopwords  # import stopwords from nltk corpus
from nltk.stem.snowball import FrenchStemmer  # import the French stemming library
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from sklearn.feature_extraction.text import CountVectorizer

import spacy
nlp = spacy.load('fr_core_news_md')

from string import punctuation
#has been tried but is not a good lemmatizer
#from cltk.lemmatize.french.lemma import LemmaReplacer
#from cltk.corpus.utils.importer import CorpusImporter


"""This function takes path to a pdf and returns a string of the whole document
    This method is the old one we were using. It caused some errors in the pdf conversion, so we decided to 
    switch to another the other version (pdf_to_text2)
"""
def pdf_to_text(pdf_path):
    pdf_manag = PDFResourceManager()
    string = StringIO()
    codec='utf-8'
    laparams = LAParams()
    device = TextConverter(pdf_manag, string, laparams=laparams)
    case = open(pdf_path, 'rb')
    interpreter = PDFPageInterpreter(pdf_manag, device)
    password=""
    maxpages=0
    caching=True
    pagenos=set()
    
    for page in PDFPage.get_pages(case, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
        interpreter.process_page(page)
    text = string.getvalue()
    
    case.close()
    device.close()
    string.close()
    return text

"""new method to convert from pdf, it seemed like it was doing a better job than the other one (which was raising these
issues with too many spaces"""
def pdftotext2(pdf_path):
    # Load your PDF
    with open(pdf_path, "rb") as f:
        pdf = pdftotext.PDF(f)

    # How many pages?
    #print(len(pdf))

    # Iterate over all the pages
    # for page in pdf:
    #   print(page)

    # Read some individual pages
    # print(pdf[0])
    # print(pdf[1])

    # Read all the text into one string
    one_string = "\n\n".join(pdf)
    # print("\n\n".join(pdf))

    return one_string

def get_tokens(raw,encoding='utf8'):
    #get the nltk tokens from a text
    tokens = nltk.word_tokenize(raw) #tokenize the raw UTF-8 text
    return tokens


def get_nltk_text(raw,encoding='utf8'):
    #create an nltk text using the passed argument (raw) after filtering out the commas
    #turn the raw text into an nltk text object
    no_commas = re.sub(r'[.|,|\']',' ', raw) #filter out all the commas, periods, and appostrophes using regex
    tokens = nltk.word_tokenize(no_commas) #generate a list of tokens from the raw text
    text=nltk.Text(tokens,encoding) #create a nltk text from those tokens
    return text


def get_stopswords(type="veronis"):
    # returns the veronis stopwords in unicode, or if any other value is passed, it returns the default nltk french stopwords
    if type == "veronis":
        # VERONIS STOPWORDS Jean Veronis is a French Linguist
        raw_stopword_list = ["Ap.", "Apr.", "GHz", "MHz", "USD", "a", "afin", "ah", "ai", "aie", "aient", "aies", "ait",
                             "alors", "après", "as", "attendu", "au", "au-delà", "au-devant", "aucun", "aucune",
                             "audit", "auprès", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras",
                             "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autour", "autre", "autres",
                             "autrui", "aux", "auxdites", "auxdits", "auxquelles", "auxquels", "avaient", "avais",
                             "avait", "avant", "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons",
                             "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était", "car", "ce",
                             "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là",
                             "celui", "celui-ci", "celui-là", "celà", "cent", "cents", "cependant", "certain",
                             "certaine", "certaines", "certains", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là",
                             "cf.", "cg", "cgr", "chacun", "chacune", "chaque", "chez", "ci", "cinq", "cinquante",
                             "cinquante-cinq", "cinquante-deux", "cinquante-et-un", "cinquante-huit", "cinquante-neuf",
                             "cinquante-quatre", "cinquante-sept", "cinquante-six", "cinquante-trois", "cl", "cm",
                             "cm²", "comme", "contre", "d", "d'", "d'après", "d'un", "d'une", "dans", "de", "depuis",
                             "derrière", "des", "desdites", "desdits", "desquelles", "desquels", "deux", "devant",
                             "devers", "dg", "différentes", "différents", "divers", "diverses", "dix", "dix-huit",
                             "dix-neuf", "dix-sept", "dl", "dm", "donc", "dont", "douze", "du", "dudit", "duquel",
                             "durant", "dès", "déjà", "e", "eh", "elle", "elles", "en", "en-dehors", "encore", "enfin",
                             "entre", "envers", "es", "est", "et", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse",
                             "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eûmes", "eût", "eûtes", "f",
                             "fait", "fi", "flac", "fors", "furent", "fus", "fusse", "fussent", "fusses", "fussiez",
                             "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gr", "h", "ha", "han", "hein", "hem",
                             "heu", "hg", "hl", "hm", "hm³", "holà", "hop", "hormis", "hors", "huit", "hum", "hé", "i",
                             "ici", "il", "ils", "j", "j'", "j'ai", "j'avais", "j'étais", "jamais", "je", "jusqu'",
                             "jusqu'au", "jusqu'aux", "jusqu'à", "jusque", "k", "kg", "km", "km²", "l", "l'", "l'autre",
                             "l'on", "l'un", "l'une", "la", "laquelle", "le", "lequel", "les", "lesquelles", "lesquels",
                             "leur", "leurs", "lez", "lors", "lorsqu'", "lorsque", "lui", "lès", "m", "m'", "ma",
                             "maint", "mainte", "maintes", "maints", "mais", "malgré", "me", "mes", "mg", "mgr", "mil",
                             "mille", "milliards", "millions", "ml", "mm", "mm²", "moi", "moins", "mon", "moyennant",
                             "mt", "m²", "m³", "même", "mêmes", "n", "n'avait", "n'y", "ne", "neuf", "ni", "non",
                             "nonante", "nonobstant", "nos", "notre", "nous", "nul", "nulle", "nº", "néanmoins", "o",
                             "octante", "oh", "on", "ont", "onze", "or", "ou", "outre", "où", "p", "par", "par-delà",
                             "parbleu", "parce", "parmi", "pas", "passé", "pendant", "personne", "peu", "plus",
                             "plus_d'un", "plus_d'une", "plusieurs", "pour", "pourquoi", "pourtant", "pourvu", "près",
                             "puisqu'", "puisque", "q", "qu", "qu'", "qu'elle", "qu'elles", "qu'il", "qu'ils", "qu'on",
                             "quand", "quant", "quarante", "quarante-cinq", "quarante-deux", "quarante-et-un",
                             "quarante-huit", "quarante-neuf", "quarante-quatre", "quarante-sept", "quarante-six",
                             "quarante-trois", "quatorze", "quatre", "quatre-vingt", "quatre-vingt-cinq",
                             "quatre-vingt-deux", "quatre-vingt-dix", "quatre-vingt-dix-huit", "quatre-vingt-dix-neuf",
                             "quatre-vingt-dix-sept", "quatre-vingt-douze", "quatre-vingt-huit", "quatre-vingt-neuf",
                             "quatre-vingt-onze", "quatre-vingt-quatorze", "quatre-vingt-quatre", "quatre-vingt-quinze",
                             "quatre-vingt-seize", "quatre-vingt-sept", "quatre-vingt-six", "quatre-vingt-treize",
                             "quatre-vingt-trois", "quatre-vingt-un", "quatre-vingt-une", "quatre-vingts", "que",
                             "quel", "quelle", "quelles", "quelqu'", "quelqu'un", "quelqu'une", "quelque", "quelques",
                             "quelques-unes", "quelques-uns", "quels", "qui", "quiconque", "quinze", "quoi", "quoiqu'",
                             "quoique", "r", "revoici", "revoilà", "rien", "s", "s'", "sa", "sans", "sauf", "se",
                             "seize", "selon", "sept", "septante", "sera", "serai", "seraient", "serais", "serait",
                             "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "si", "sinon", "six",
                             "soi", "soient", "sois", "soit", "soixante", "soixante-cinq", "soixante-deux",
                             "soixante-dix", "soixante-dix-huit", "soixante-dix-neuf", "soixante-dix-sept",
                             "soixante-douze", "soixante-et-onze", "soixante-et-un", "soixante-et-une", "soixante-huit",
                             "soixante-neuf", "soixante-quatorze", "soixante-quatre", "soixante-quinze",
                             "soixante-seize", "soixante-sept", "soixante-six", "soixante-treize", "soixante-trois",
                             "sommes", "son", "sont", "sous", "soyez", "soyons", "suis", "suite", "sur", "sus", "t",
                             "t'", "ta", "tacatac", "tandis", "te", "tel", "telle", "telles", "tels", "tes", "toi",
                             "ton", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente",
                             "trente-cinq", "trente-deux", "trente-et-un", "trente-huit", "trente-neuf",
                             "trente-quatre", "trente-sept", "trente-six", "trente-trois", "trois", "très", "tu", "u",
                             "un", "une", "unes", "uns", "v", "vers", "via", "vingt", "vingt-cinq", "vingt-deux",
                             "vingt-huit", "vingt-neuf", "vingt-quatre", "vingt-sept", "vingt-six", "vingt-trois",
                             "vis-à-vis", "voici", "voilà", "vos", "votre", "vous", "w", "x", "y", "z", "zéro", "à",
                             "ç'", "ça", "ès", "étaient", "étais", "était", "étant", "étiez", "étions", "été", "étée",
                             "étées", "étés", "êtes", "être", "ô"]
    else:
        # get French stopwords from the nltk kit
        raw_stopword_list = stopwords.words('french')  # create a list of all French stopwords
    # stopword_list = [word.decode('utf8') for word in raw_stopword_list] #make to decode the French stopwords as unicode objects rather than ascii

    # also add the punctuation signs to the list of stopwords
    signs = list(set(punctuation))
    #this is a list of keywords that we found to be quite prominent in the final keywords analysis. As they
    #dont have an actual meaning, but are not considered 'punctuation' or 'stopwords', we manually add them to the
    # list of words to retrieve
    stopwords_law = ["xiii", "xiiir", "viii", "xi", "vi", "viiir", "xiiie", "xxx"]

    raw_stopword_list = raw_stopword_list + signs + stopwords_law
    return raw_stopword_list



def filter_stopwords(text,stopword_list):
    #normalizes the words by turning them all lowercase and then filters out the stopwords
    words=[str(w).lower() for w in text] #normalize the words in the text, making them all lowercase
    #filtering stopwords
    filtered_words = [] #declare an empty list to hold our filtered words
    for word in words: #iterate over all words from the text
        if word not in stopword_list and word.isalpha() and len(word) > 1: #only add words that are not in the French stopwords list, are alphabetic, and are more than 1 character
            filtered_words.append(word) #add word to filter_words list if it meets the above conditions
    filtered_words.sort() #sort filtered_words list
    return filtered_words

def stem_words(words):
    #stems the word list using the French Stemmer
    #stemming words
    stemmed_words = [] #declare an empty list to hold our stemmed words
    stemmer = FrenchStemmer() #create a stemmer object in the FrenchStemmer class
    for word in words:
        stemmed_word=stemmer.stem(word) #stem the word
        stemmed_words.append(stemmed_word) #add it to our stemmed word list
    stemmed_words.sort() #sort the stemmed_words
    return stemmed_words



def sort_dictionary(dictionary):
    #returns a sorted dictionary (as tuples) based on the value of each key
    return sorted(dictionary.items(), key=lambda x: x[1], reverse=True)


def print_sorted_dictionary(tuple_dict):
    #print the results of sort_dictionary
    for tup in tuple_dict:
        print (str(tup[1])[0:10] + '\t\t' + str(tup[0]))


def clean_vocabulary(df, stopwords):
    # clean all cases texts
    for index, case in df.iterrows():
        print(index)
        case_text = case['case_text']
        if isinstance(case_text, str):
            lem_case = nlp(case_text)
            lem_words = list()
            for word in lem_case:
                lem_words.append(word.lemma_)
            lem_words = filter_stopwords(lem_words, stopwords)
            words_str = ' '.join(map(str, lem_words))
            df.loc[index, 'case_text'] = words_str
        else:
            df.loc[index, 'case_text'] = ''
    return df


def retrieve_stopwords_numbers(df, stopwords):
    for index, case in df.iterrows():
        print(index)
        case_text = case['case_text']
        if isinstance(case_text, str):
            text = nlp(case_text)
        # for w in text:
        #    print(w)
            filtered_words = filter_stopwords(text, stopwords)
            words_str = ' '.join(map(str, filtered_words))
        else:
            words_str=''
            
        df.loc[index, 'case_text'] = words_str
    return df

# ## Create the second version of the data set with the lemmatized texts
def create_lemmatized_datasets(case_topic_termdoc_env, case_topic_termdoc_non_env):


    stopwords = get_stopswords()
    #env data set
    clean_env = clean_vocabulary(case_topic_termdoc_env, stopwords)
    #non env data set
    clean_non_env = clean_vocabulary(case_topic_termdoc_non_env, stopwords)
    return clean_env, clean_non_env

def create_lemmatized_dataset(case_topic_termdoc):
    stopwords = get_stopswords()
    # env data set
    clean = clean_vocabulary(case_topic_termdoc, stopwords)

    return clean

# Create the version of the data sets with non-lemmatized vocabulary, but without numbers and stopwords
def create_no_stopwords_datasets(case_topic_termdoc_env, case_topic_termdoc_non_env):

    stopwords = get_stopswords()
    #env data set
    no_stopwords_env = retrieve_stopwords_numbers(case_topic_termdoc_env, stopwords)
    #non env data set
    no_stopwords_non_env = retrieve_stopwords_numbers(case_topic_termdoc_non_env, stopwords)

    return no_stopwords_env, no_stopwords_non_env

def create_no_stopwords_dataset(case_topic_termdoc):

    stopwords = get_stopswords()
    #env data set
    no_stopwords = retrieve_stopwords_numbers(case_topic_termdoc, stopwords)

    return no_stopwords

"""This method returs the year of a case, given its full path """
def get_year_from_path(path):
    #print(path)
    path = path[0:len(path)-4]
    paths = path.split('/')
    path=paths[len(paths)-1]
    year = paths[2]
    return year

"""This is the new method that divided the cases into the environmental or non-environmental data sets.
it return 2 tables where the colun names are the most specific topic names for all the cases under them.

Function that creates the topics_env and topics_non_env matrices

Methodology :
1. go through all csv files that contain the case_numbers (scraped per topic from Juridict)
2. all 'environmental cases' will be in 'Aménagement de la nature, de l'environnement et urbanisme'. Hence, if 'nature' is not in the topic name, we directly save the cases in the topics_non_env matrix
3. if 'nature' is in, we save all cases as environmental except from those who are in 'non_env_to_substract' list of keywords. These keywords have been manually double-checked by matthias as being non-environmental.
"""
def merge_sub_db_2():

    non_env_to_substract = ['Trafic aérien',
                            'Camping',
                            'mines',
                            'Chasse/Généralités',
                            'Chasse/Loi du 28 février 1882',
                            'Chasse/Permis de chasse',
                            'Code de',
                            'Déchets',
                            "eau navigables",
                            "eau non navigables/",
                            'Eaux/Dispositions relatives à la lutte contre les inondations',
                            "Eaux/Distribution",
                            "Eaux/Organismes",
                            'Eaux/Subventions',
                            'Logement',
                            'Monuments et sites',
                            'Remembrement',
                            'Urbanisme et aménagement du territoire',
                            'Voirie',
                            'Varia']

    #merge all sub-databases
    topics_non_env=pd.DataFrame()
    topics_env = pd.DataFrame()
    db_dir = 'data/all_topics_csv/'
    for topic_db in os.listdir(db_dir):
        print(topic_db)

        if '.csv' in topic_db:
            db = pd.read_csv(str(db_dir) + str(topic_db))
            db = db.loc[(db != 0).any(1)]

            del db['Unnamed: 0']

            #list_of_col_to_substract = list()

            env_columns =list()
            non_env_columns=list()

            for col in db.columns:
                col_name = str(col)

                # this is a column that is should be non-environmental
                if ' environnement et nature' not in col_name:
                    non_env_columns.append(col_name)

                else:
                    non_env = False
                    for to_substract in non_env_to_substract:
                        if to_substract in col_name:
                            non_env=True

                    if non_env == True:
                        non_env_columns.append(col_name)
                    else:
                        #print('add to env : '+str(col_name))
                        env_columns.append(col_name)


                    #if we have not yet appended it
                    #topics_non_env = topics_non_env.append(new_df)


                        # print(topics_non_env.shape)
            #print('environmental minus non-environmental columns')
            # delete all columns that were actually non-environmental

            for col in non_env_columns:
                #print('NON ENV columns: ' + str(col))
                #del db[col]
                #topics_non_env = topics_non_env.append(db[col])
                topics_non_env[col] = db[col]

            for col in env_columns:
                #print('ENV columns : '+str(col))
                #topics_env = topics_env.append(db[col])
                topics_env[col] = db[col]




    topics_env = topics_env.fillna(0)
    topics_non_env = topics_non_env.fillna(0)


    """print("FINAL COLUMNS PRINTING")
    print("ENV COLUMNS")
    for env_col in env_columns:
        print('ENV columns : '+str(col))
    print("NON ENV COLUMNS")
    for non_env in topics_non_env.columns:
        print(non_env)"""

    print('FINAL SHAPES : ')
    print(topics_non_env.shape)
    print(topics_env.shape)
    return topics_env, topics_non_env




"""Creates the case_term_doc_matrix_env and case_term_doc_matrix_non_env
1. retrieve all the case numbers from the given matrix (either topics_env or topics_non_env)
2. create an entry for each case withe case_number, topic, link to vocabulary file (currently null)"""
def create_case_topic_df(topics_env):
    print("start creating case termdoc matrix")
    counter=0
    cases_count=0

    case_topic_termdoc = pd.DataFrame(columns=['case_number', 'topic', 'term_doc_matrix'])

    for ind, column in enumerate(topics_env.columns):
        #print(ind)
        #print(column)
        case_numbers = topics_env[column]
        case_numbers= case_numbers[case_numbers!=0]
        case_numbers = [nb for nb in case_numbers if nb]
        for case_number in case_numbers:
            #print(case_number)
            if case_number is not None:
                case_topic_termdoc = case_topic_termdoc.append({'case_number':int(case_number), 'topic':column,'term_doc_matrix':0}, ignore_index=True)

        cases_count = cases_count+len(case_numbers)

    print('case topic shape: '+str(case_topic_termdoc.shape))

    return case_topic_termdoc

"""This function fills the term_doc_matrix column from the case_topic_termdoc matrix. It also adds a 'text' that we realized would be useful in implementing the machine learning algorithm
1. for each year, go through all cases and retrieve their case number
2. if the case_number is also in the case_topic_termdoc, add the path to its vocabulary to the termdoc column. then, change its pdf to text, and add the full string to the 'text' column
"""
def link_juri_and_full_dataset(case_topic_termdoc):

    error_count=0

    case_topic_termdoc.term_doc_matrix = case_topic_termdoc.term_doc_matrix.astype(str)
    print("start filling case termdoc matrix")
    #add a 'text' column, this will facilitate our use of the `CountVectorizer sklearn feature that takes in a list of text
    texts = [None] * len(case_topic_termdoc)
    case_topic_termdoc['case_text'] = texts

    #go through the whole archive and whenever we find a case that is present in the dataframe, append its term_doc matrix to the one above
    #archive_dir = "/Users/marion1meyers/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive_New/"
    archive_dir = "data/full_dataset/"
    true_counter =0
    false_counter=0
    for year in os.listdir(archive_dir):
        if year=='.DS_Store':
            continue
        print(year)
        french_dir = archive_dir+str(year)+"/french/"

        for case in os.listdir(french_dir):
            print(case)
            if 'pdf' in case:
                print('pdf in it')
                if 'trad' in case:
                    case2=str(case)
                    case2 = case2[0:len(case2)-9]
                else:
                    case2=case[0:len(case)-4]
                parts = case2.split('_')
                case_number=parts[2]
                case_number = ''.join(c for c in case_number if c.isdigit())
                #print(case_number)
                case_number = int(case_number)
                print('case number '+str(case_number))


                #add the corresponding csv file path to the case_n, topic database
                #if this is more than one row than this means that the same case can be classified in different topics
                if case_number in case_topic_termdoc.case_number.values:
                    true_counter = true_counter+1

                    index = case_topic_termdoc.index[case_topic_termdoc['case_number']==case_number][0]

                    case_topic_termdoc.at[index, 'term_doc_matrix'] = french_dir + str(case_number) + ".csv"

                    #case_text = pdf_to_text(french_dir + str(case))

                    #some pdf files are damaged, we therefore need to add a try/catch statement
                    try:
                        case_text = pdftotext2(french_dir + str(case))
                    except:
                        print("ERROR IN FILE ")
                        case_text =""
                        error_count+=1
                    case_topic_termdoc.at[index, 'case_text'] = case_text

                else:
                    false_counter = false_counter+1

        print('true counter: '+str(true_counter))
        print('false counter: '+str(false_counter))
    return case_topic_termdoc, error_count

def create_unlabelled_dataset(archive_dir):
    case_topic_termdoc = pd.DataFrame(columns=['case_number', 'topic', 'term_doc_matrix','year', 'case_text'])


    # go through the whole archive and whenever we find a case that is present in the dataframe, append its term_doc matrix to the one above
    # archive_dir = "/Users/marion1meyers/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive_New/"

    for year in os.listdir(archive_dir):
        if year == '.DS_Store':
            continue
        print(year)
        french_dir = archive_dir + str(year) + "/french/"

        for case in os.listdir(french_dir):
            print(case)
            if 'pdf' in case:
                print('pdf in it')
                if 'trad' in case:
                    case2 = str(case)
                    case2 = case2[0:len(case2) - 9]
                else:
                    case2 = case[0:len(case) - 4]
                parts = case2.split('_')
                case_number = parts[2]
                case_number = ''.join(c for c in case_number if c.isdigit())
                # print(case_number)
                case_number = int(case_number)
                print('case number ' + str(case_number))

                try:
                    #case_text = pdf_to_text(french_dir + str(case))
                    case_text = pdftotext2(french_dir + str(case))
                except:
                    case_text=""

                case_topic_termdoc = case_topic_termdoc.append({"case_number":case_number, "topic":0, "term_doc_matrix":french_dir + str(case_number) + ".csv", "year":get_year_from_path(french_dir + str(case_number) ),"case_text":case_text}, ignore_index=True)


    return case_topic_termdoc

"""


#Code that I run on the new google cloud (marionlawtechlab) to re-do the entire data set and compare with the new results
#1. recreate the env and non-env topic, with adding 'Varia' in the list of 'to be excluded from the environmental topics'
#2. from the topics, create the case_termdoc matrices
#3. fill these in with both the path to the matrix, as well as the case_text. one matrix with the text, one with the lemmatized version of it
"""

"""9th of June: Re-creating the labelled data set. i have change the pdf to text method i was using in order to (hopefully)
fix all the issues that were leading to some texts being translating into a list of super random words (heure, gramme...). 
I was also manually able to correct for these cases (using the new pdf to text method) but i realized that there were still
many pdfs that actually raised errors when being transformed. What i was previously doing was simply to discard these texts and 
not have them in the data set. However,  i now realize that maybe by using the new pdf to text method, these texts will no longer
raise a problem. If this method works out better than the old (it should), i will also use it to create the final data set of all 
the unlabelled cases. """

#1.TOPICS
"""
topics_env, topics_non_env = merge_sub_db_2()
topics_env.to_csv('topics_env_22_june.csv', index=False)
topics_non_env.to_csv('topics_non_env_22_june.csv', index=False)

#print('topics saved')
"""

topics_env = pd.read_csv('topics_env_22_june.csv')
topics_non_env = pd.read_csv('topics_non_env_22_june.csv')


#2. CASE TERM DOC MATRICES:
#i used to say keep first when duplicates, I have deleted this line so that the duplicates are still there. I will take care of that later.
case_topic_env = create_case_topic_df(topics_env)
case_topic_non_env = create_case_topic_df(topics_non_env)
case_topic_env.to_csv('case_topic_env_not_full_22_june.csv', index=False)
case_topic_non_env.to_csv('case_topic_non_env_not_full_22_june.csv', index=False)
print('case termdoc created')

case_topic_env = pd.read_csv("case_topic_env_not_full_22_june.csv")
case_topic_non_env = pd.read_csv("case_topic_non_env_not_full_22_june.csv")

#3. FIll them in with the full text
case_topic_env_full, env_error_count = link_juri_and_full_dataset(case_topic_env)
case_topic_non_env_full, non_env_error_count = link_juri_and_full_dataset(case_topic_non_env)
case_topic_env_full.to_csv('case_topic_env_full_22_june.csv', index=False)
case_topic_non_env_full.to_csv('case_topic_non_env_full_22_june.csv', index=False)
print('full case termdoc matrices saved')
print("number of errors when converting pdf into texts")
print('for env cases : '+str(env_error_count))
print('for non env cases: '+str(non_env_error_count))



lemm_env, lemm_non_env = create_lemmatized_datasets(case_topic_env_full, case_topic_non_env_full)
lemm_env.to_csv("case_topic_env_lemmatized_22_june.csv", index=False)
lemm_non_env.to_csv("case_topic_non_env_lemmatized_22_june.csv", index=False)


ns_env, ns_non_env = create_no_stopwords_datasets(case_topic_env_full, case_topic_non_env_full)
ns_env.to_csv("case_topic_env_no_stopwords_22_june.csv", index=False)
ns_non_env.to_csv("case_topic_non_env_no_stopwords_22_june.csv", index=False)


#CODE to create the data set of the entire set of documents (unlabelled data)
archive_dir ="data/full_dataset/"
#archive_dir = "/users/marion1meyers/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive_New/"

dataset = create_unlabelled_dataset(archive_dir)
dataset.to_csv("unlabelled_text_dataset.csv", index=False)
print('unlabelled data set created')

#dataset = pd.read_csv("data/full_text_dataset.csv")


lemmatized_dataset = create_lemmatized_dataset(dataset)
lemmatized_dataset.to_csv("unlabelled_lemmatized_dataset.csv", index=False)
print('lemmatized dataset created ')

no_stopwords_dataset = create_no_stopwords_dataset(dataset)
no_stopwords_dataset.to_csv("unlabelled_no_stopwords_dataset.csv", index=False)
print('no stopwords data set created')


#get the number of rows. this is equal to the total number of cases we have webscraped on Raad van Staat website
#print(dataset.shape)



"""In the week before submitting the research proposal, i realized that there were a lot of cases that were actually misclassified. 
I think the error was in the deletion of the non-env columns (see method above). I am re-trying.
"""
#topics_env, topics_non_env = merge_sub_db()
#topics_env.to_csv('topics_env_15_june.csv', index=False)
#topics_non_env.to_csv('topics_non_env_15_june.csv', index=False)

#print('topics saved')


