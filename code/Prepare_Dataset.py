#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re  # import the regular expressions library
from io import StringIO

# imports
import nltk  # import the natural language toolkit library
import numpy as np
import pandas as pd
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from matplotlib import pyplot as plt
from nltk.corpus import stopwords  # import stopwords from nltk corpus
from nltk.stem.snowball import FrenchStemmer  # import the French stemming library
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from sklearn.feature_extraction.text import CountVectorizer


#has been tried but is not a good lemmatizer
#from cltk.lemmatize.french.lemma import LemmaReplacer
#from cltk.corpus.utils.importer import CorpusImporter



# In[2]:


#This should just be run once for every copora that you want to be able to lemmatize on. CLTK

#corpus_importer = CorpusImporter('french')  # e.g., or CorpusImporter('latin')

#list the french corpora available and then download all of them. 
#corpus_importer.list_corpora

#corpus_importer.import_corpus('french_data_cltk')


# In[3]:

"""This function takes path to a pdf and returns a string of the whole document"""
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

#case_ex1 = pdf_to_text("C:/Users/mario/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive/1995/flemish/51000_000_51060.pdf")
#case_ex2 = pdf_to_text("C:/Users/mario/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive/1995/flemish/51000_000_51061.pdf")
#case_ex3 = pdf_to_text("C:/Users/mario/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive/1995/flemish/51000_000_51062.pdf")
#case_ex4 = pdf_to_text("C:/Users/mario/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive/1995/flemish/51000_000_51063.pdf")
#case_ex5 = pdf_to_text("C:/Users/mario/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive/1995/flemish/51000_000_51064.pdf")

#path = "C:/Users/mario/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive_Small/Flemish/1995/"
#path = "C:/Users/mario/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive/1995/Flemish/"
#cases = []
# r=root, d=directories, f = files
#for r, d, f in os.walk(path):
 #   for file in f:
  #      case = pdf_to_text(os.path.join(r, file))
   #     cases.append(case)


#print(len(cases))   


# ## Clean the text : word stemming etc. 

# In[4]:


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


# In[6]:


def get_stopswords(type="veronis"):
    #returns the veronis stopwords in unicode, or if any other value is passed, it returns the default nltk french stopwords
    if type=="veronis":
        #VERONIS STOPWORDS Jean Veronis is a French Linguist
        raw_stopword_list = ["Ap.", "Apr.", "GHz", "MHz", "USD", "a", "afin", "ah", "ai", "aie", "aient", "aies", "ait", "alors", "après", "as", "attendu", "au", "au-delà", "au-devant", "aucun", "aucune", "audit", "auprès", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autour", "autre", "autres", "autrui", "aux", "auxdites", "auxdits", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons", "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était", "car", "ce", "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là", "celà", "cent", "cents", "cependant", "certain", "certaine", "certaines", "certains", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "cf.", "cg", "cgr", "chacun", "chacune", "chaque", "chez", "ci", "cinq", "cinquante", "cinquante-cinq", "cinquante-deux", "cinquante-et-un", "cinquante-huit", "cinquante-neuf", "cinquante-quatre", "cinquante-sept", "cinquante-six", "cinquante-trois", "cl", "cm", "cm²", "comme", "contre", "d", "d'", "d'après", "d'un", "d'une", "dans", "de", "depuis", "derrière", "des", "desdites", "desdits", "desquelles", "desquels", "deux", "devant", "devers", "dg", "différentes", "différents", "divers", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dl", "dm", "donc", "dont", "douze", "du", "dudit", "duquel", "durant", "dès", "déjà", "e", "eh", "elle", "elles", "en", "en-dehors", "encore", "enfin", "entre", "envers", "es", "est", "et", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eûmes", "eût", "eûtes", "f", "fait", "fi", "flac", "fors", "furent", "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gr", "h", "ha", "han", "hein", "hem", "heu", "hg", "hl", "hm", "hm³", "holà", "hop", "hormis", "hors", "huit", "hum", "hé", "i", "ici", "il", "ils", "j", "j'", "j'ai", "j'avais", "j'étais", "jamais", "je", "jusqu'", "jusqu'au", "jusqu'aux", "jusqu'à", "jusque", "k", "kg", "km", "km²", "l", "l'", "l'autre", "l'on", "l'un", "l'une", "la", "laquelle", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "lez", "lors", "lorsqu'", "lorsque", "lui", "lès", "m", "m'", "ma", "maint", "mainte", "maintes", "maints", "mais", "malgré", "me", "mes", "mg", "mgr", "mil", "mille", "milliards", "millions", "ml", "mm", "mm²", "moi", "moins", "mon", "moyennant", "mt", "m²", "m³", "même", "mêmes", "n", "n'avait", "n'y", "ne", "neuf", "ni", "non", "nonante", "nonobstant", "nos", "notre", "nous", "nul", "nulle", "nº", "néanmoins", "o", "octante", "oh", "on", "ont", "onze", "or", "ou", "outre", "où", "p", "par", "par-delà", "parbleu", "parce", "parmi", "pas", "passé", "pendant", "personne", "peu", "plus", "plus_d'un", "plus_d'une", "plusieurs", "pour", "pourquoi", "pourtant", "pourvu", "près", "puisqu'", "puisque", "q", "qu", "qu'", "qu'elle", "qu'elles", "qu'il", "qu'ils", "qu'on", "quand", "quant", "quarante", "quarante-cinq", "quarante-deux", "quarante-et-un", "quarante-huit", "quarante-neuf", "quarante-quatre", "quarante-sept", "quarante-six", "quarante-trois", "quatorze", "quatre", "quatre-vingt", "quatre-vingt-cinq", "quatre-vingt-deux", "quatre-vingt-dix", "quatre-vingt-dix-huit", "quatre-vingt-dix-neuf", "quatre-vingt-dix-sept", "quatre-vingt-douze", "quatre-vingt-huit", "quatre-vingt-neuf", "quatre-vingt-onze", "quatre-vingt-quatorze", "quatre-vingt-quatre", "quatre-vingt-quinze", "quatre-vingt-seize", "quatre-vingt-sept", "quatre-vingt-six", "quatre-vingt-treize", "quatre-vingt-trois", "quatre-vingt-un", "quatre-vingt-une", "quatre-vingts", "que", "quel", "quelle", "quelles", "quelqu'", "quelqu'un", "quelqu'une", "quelque", "quelques", "quelques-unes", "quelques-uns", "quels", "qui", "quiconque", "quinze", "quoi", "quoiqu'", "quoique", "r", "revoici", "revoilà", "rien", "s", "s'", "sa", "sans", "sauf", "se", "seize", "selon", "sept", "septante", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "si", "sinon", "six", "soi", "soient", "sois", "soit", "soixante", "soixante-cinq", "soixante-deux", "soixante-dix", "soixante-dix-huit", "soixante-dix-neuf", "soixante-dix-sept", "soixante-douze", "soixante-et-onze", "soixante-et-un", "soixante-et-une", "soixante-huit", "soixante-neuf", "soixante-quatorze", "soixante-quatre", "soixante-quinze", "soixante-seize", "soixante-sept", "soixante-six", "soixante-treize", "soixante-trois", "sommes", "son", "sont", "sous", "soyez", "soyons", "suis", "suite", "sur", "sus", "t", "t'", "ta", "tacatac", "tandis", "te", "tel", "telle", "telles", "tels", "tes", "toi", "ton", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "trente-cinq", "trente-deux", "trente-et-un", "trente-huit", "trente-neuf", "trente-quatre", "trente-sept", "trente-six", "trente-trois", "trois", "très", "tu", "u", "un", "une", "unes", "uns", "v", "vers", "via", "vingt", "vingt-cinq", "vingt-deux", "vingt-huit", "vingt-neuf", "vingt-quatre", "vingt-sept", "vingt-six", "vingt-trois", "vis-à-vis", "voici", "voilà", "vos", "votre", "vous", "w", "x", "y", "z", "zéro", "à", "ç'", "ça", "ès", "étaient", "étais", "était", "étant", "étiez", "étions", "été", "étée", "étées", "étés", "êtes", "être", "ô"]
    else:
        #get French stopwords from the nltk kit
        raw_stopword_list = stopwords.words('french') #create a list of all French stopwords
    #stopword_list = [word.decode('utf8') for word in raw_stopword_list] #make to decode the French stopwords as unicode objects rather than ascii
    return raw_stopword_list


# In[7]:


def get_stopswords_dutch():
    with open("C:/Users/mario/Documents/LawTechLab/environmental-args-belgian-law/stopwords-nl-master/stopwords-nl.txt") as f:
        stop_words = f.readlines()
    #print(content)
    stop_words = [x.strip() for x in stop_words] 
    return stop_words


# In[8]:


def filter_stopwords(text,stopword_list):
    #normalizes the words by turning them all lowercase and then filters out the stopwords
    words=[w.lower() for w in text] #normalize the words in the text, making them all lowercase
    #filtering stopwords
    filtered_words = [] #declare an empty list to hold our filtered words
    for word in words: #iterate over all words from the text
        if word not in stopword_list and word.isalpha() and len(word) > 1: #only add words that are not in the French stopwords list, are alphabetic, and are more than 1 character
            filtered_words.append(word) #add word to filter_words list if it meets the above conditions
    filtered_words.sort() #sort filtered_words list
    return filtered_words


# In[9]:


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


# In[10]:


def lemmatize_words(words):
    lem_words=[]
    lemmatizer = FrenchLefffLemmatizer()
    
    for word in words: 
        lem_word=lemmatizer.lemmatize(word)
        #lem_word_2 = lemmatizer2.lemmatize(word)
        
        lem_words.append(lem_word)
        print('w: '+str(word)+', l : '+str(lem_word))
    #lem_words.sort()
    #This lemmatizer has been tried but is not good: many words are lemmatized to None as they are unknown by the lemmatizer. 
    #print('lemmatize 2 : ')
    #lemmatizer2 = LemmaReplacer()
    #print(lemmatizer2.lemmatize(words))                  
    return lem_words


# In[11]:


def sort_dictionary(dictionary):
    #returns a sorted dictionary (as tuples) based on the value of each key
    return sorted(dictionary.items(), key=lambda x: x[1], reverse=True)


# In[12]:


def print_sorted_dictionary(tuple_dict):
    #print the results of sort_dictionary
    for tup in tuple_dict:
        print (str(tup[1])[0:10] + '\t\t' + str(tup[0]))




# In[ ]:
""" This function creates a vocabulary matrix per year by doing the following: 
1. retrieves all cases as pdf for a certain year
2. translates them into strings (after having filtered out the stop words)
3. creates a vocabulary how of all cases for a certain year using a CountVectorizer
4. saves this vocabulary as a matrix"""
def create_year_vocab():
    years = range(1995,2019)
    for year in years:
        print(year)
        path = "C:/Users/mario/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive/"+str(year)+"/flemish/"
        cases = []
        french_stopwords = get_stopswords()
        dutch_stopwords = get_stopswords_dutch()
        #r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                try:
                    case = pdf_to_text(os.path.join(r, file))
                    nltk_text = get_nltk_text(case)
                    words = filter_stopwords(nltk_text,dutch_stopwords)
                    #print('all words but stop words')
                    #print(words)
                    #words = lemmatize_words(words)
                    #print("lemmatized words")
                    #print(words)
                    #words = stem_words(words)
                    #print('stemmed words')
                    #print(words)
                    words_str = ' '.join(map(str, words))
                    cases.append(words_str)
                except Exception as e:
                    print(e)
        vect = CountVectorizer(ngram_range=(1,1))
        #cases = [words_str, words_str, words_str]
        dtm = vect.fit_transform(cases)
        print(type(dtm))
        feature_names = vect.get_feature_names()
        term_doc_matrix = pd.DataFrame(dtm.toarray(), columns=feature_names)
        term_doc_matrix.to_csv (r''+str(path)+'term_doc.csv', index = None, header=True)



""" This function creates a vocabulary matrix per case by doing the following: 
1. retrieves a case as pdf, and retrieve its actual case_number
2. translates it into strings (after having filtered out the stop words)
3. creates a vocabulary for the case using a CountVectorizer
4. saves this vocabulary as a matrix with the case_number as file name"""
def create_indiv_vocab():
    #create term doc for each file
    """NOT DONE STARTING IN 1999"""
    years = range(2008,2020)
    for year in years:
        print(year)
        path = "C:/Users/mario/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive_New/"+str(year)+"/french/"
        cases = []
        french_stopwords = get_stopswords()
        #dutch_stopwords = get_stopswords_dutch()
        #r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if '.pdf' in file:
                    #try:
                    #print(file)
                    try:
                        case = pdf_to_text(os.path.join(r, file))
                    except:
                        print('error when translating to pdf')
                        continue
                    nltk_text = get_nltk_text(case)
                    words = filter_stopwords(nltk_text,french_stopwords)

                    vocabulary=list()
                    for word in words:
                        if word not in vocabulary:
                            vocabulary.append(word)
                    if len(vocabulary) == 0:
                        print(case+" had an empty vocabulary list for some reason")
                        continue
                    term_freq_matrix= pd.DataFrame(columns=vocabulary)
                    term_freq_matrix.loc[0] = np.zeros(len(vocabulary))
                    for word in words:
                        term_freq_matrix[word][0] = term_freq_matrix[word][0]+1

                    if 'trad' in file:
                        file=str(file)
                        file = file[0:len(file)-9]
                        #case_number = case[len(case)-5: len(case)]
                    else:
                        file=file[0:len(file)-4]
                    print(file)
                    parts = file.split('_')
                    case_number=parts[2]
                    case_number = ''.join(c for c in case_number if c.isdigit())
                    #print(case_number)
                    case_number = int(case_number)

                    #print(case_number)
                    term_freq_matrix.to_csv(path+str(case_number)+'.csv')
                    #print('saved csv at '+str(path)+str(case_number))

                    #except Exception as e:
                     #   print(e)


# In[52]:


#"cases = {case_ex1, case_ex2, case_ex3, case_ex4, case_ex5}
#case_as_list = case_ex.split(' ') 
#print(len(case_as_list))
#stop_words_dutch = get_stop_words('dutch')
#clean_case = [x for x in case_ex if x!=' ']

#for word in case_as_list:
#    print(word)

#vect = CountVectorizer(ngram_range=(1,1), stop_words=stop_words_dutch)
#you give the method the list of documents that creates a corpus
#TO DO : I think that it is the entire set of documents that has to be inserted here
#returns a term_document matrix where colu
# = CountVectorizer(ngram_range=(1,1))
#cases = [words_str, words_str, words_str]
#dtm = vect.fit_transform(cases)
#print(type(dtm))
#feature_names = vect.get_feature_names()
#term_doc_matrix = pd.DataFrame(dtm.toarray(), columns=feature_names)


# In[53]:


#print(term_doc_matrix.head())
#print(term_doc_matrix.head())
#term_doc_csv = term_doc_matrix.to_csv (r''+str(path)+'term_doc.csv', index = None, header=True)

#word_sums = term_doc_matrix.sum()
#print(word_sums.head())
#print(word_sums.transpose().shape)
#word_sums = word_sums.transpose()

#sorted_word_sums = word_sums.sort_values(ascending=False)
#print(sorted_word_sums)
#print(sorted_word_sums.index)

#go tthough the list, and for the forst 100 words, delete their corresponding column from the ter_doc_matrix
#for word in sorted_word_sums.index[0:100]:
#    print(word)
#    print('pre shape : '+str(term_doc_matrix.shape))
#    del term_doc_matrix[word]
#    print('post shape : '+str(term_doc_matrix.shape))

#for word in term_doc_matrix.columns:
#    print(word)
#    if word.isdigit():
#        print('delete number')
#        del term_doc_matrix[word]


# In[54]:


#print(term_doc_matrix)
#print(type(sorted_word_sums))
#print(sorted_word_sums)


# ## Analyze the presence of keywords related to the environment

# In[77]:


"""Visualize the evolution of 'environmental word' over the years
using the vocabulary matrices created per year."""
def visualize_env_words_year():
    #words that i have manually chose as 'environmental'
    env_words =['nature', 'arbre', 'environnement', 'pollution', 'air', 'parc', 'vert']
    years = list(range(1994,2019))

    env_words_per_year=list()
    env_words_df = pd.DataFrame(columns=env_words)
    env_words_doc_perc = pd.DataFrame(columns=env_words)

    for year in years:
        print(year)
        path_french = "C:/Users/mario/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive/"+str(year)+"/French/"
        term_doc_matrix = pd.read_csv(path_french+"term_doc.csv")
        print('matrix size : '+str(term_doc_matrix.shape))
        word_sums = term_doc_matrix.sum()
        sorted_word_sums = word_sums.sort_values(ascending=False)
        print(type(sorted_word_sums))

        zero_read_sums = term_doc_matrix.isin([0]).sum()
        #print('zeros sum : ')
        #print(zero_sums.sort_values(ascending=False))
        non_zeros_sums = term_doc_matrix.shape[0] - zero_sums
        #print('non zeros sums')
        #print(non_zeros_sums.sort_values(ascending=True))
        word_percentages_all = non_zeros_sums/term_doc_matrix.shape[0]
        #print('word percentages')
        #print(word_percentages_all.sort_values(ascending=True))

        #print('number of documents in the year : '+str(term_doc_matrix.shape[0]))

        env_words_count = pd.Series(name = year)
        word_percentages = pd.Series(name=year)

        for index, word in enumerate(env_words):
            #always normalize by the number of documents in the matrix
            if word in sorted_word_sums:
                env_words_count[word] = sorted_word_sums[word]/term_doc_matrix.shape[0]
                word_percentages[word] = word_percentages_all[word]
            else:
                env_words_count[word] = 0

        #ENV WORDS COUNT
        #env words count is a value computed per year and per word and corresponds to the average number of times this word is used
        #in a case of that year
        print('average number of times the word is used in a doc')
        print(env_words_count)

        print('percentage of documents that have the word in them at least once')
        print(word_percentages)

        #env_words_per_year.append(env_words_count)
        env_words_df = env_words_df.append(env_words_count)
        env_words_doc_perc = env_words_doc_perc.append(word_percentages)
        #print(env_words_df)


    #plt.bar(env_words, env_words_per_year[0])
    #plt.show()
    N = len(env_words)
    ind = np.arange(N)
    width=0.27
    print(len(years))
    print(env_words_df.shape)
    fig, ax = plt.subplots(figsize=(len(years)+1, 3))

    #ax.set_title('style: {!r}'.format(sty), color='C0')
    for word in env_words:
        ax.plot(years, env_words_df[word], label=word)
    #ax.bar(ind-width, env_words_per_year[0], width=0.2, color='b', align='center')
    #ax.bar(ind, env_words_per_year[1], width=0.2, color='g', align='center')
    #ax.bar(ind+width, env_words_per_year[2] , width=0.2, color='r', align='center')
    #ax.set_xticklabels( ('nature', 'arbre', 'environnement', 'pollution', 'air', 'parc', 'vert') )
    #ax.xaxis_date()
    ax.legend()
    plt.show()

    #plt.bar(env_words, env_words_per_year[0])
    #plt.show()
    N = len(env_words)
    ind = np.arange(N)
    width=0.27
    print(len(years))
    print(env_words_doc_perc .shape)
    fig, ax = plt.subplots(figsize=(len(years)+1, 3))

    #ax.set_title('style: {!r}'.format(sty), color='C0')
    for word in env_words:
        ax.plot(years,env_words_doc_perc [word], label=word)
    #ax.bar(ind-width, env_words_per_year[0], width=0.2, color='b', align='center')
    #ax.bar(ind, env_words_per_year[1], width=0.2, color='g', align='center')
    #ax.bar(ind+width, env_words_per_year[2] , width=0.2, color='r', align='center')
    #ax.set_xticklabels( ('nature', 'arbre', 'environnement', 'pollution', 'air', 'parc', 'vert') )
    #ax.xaxis_date()
    ax.legend()
    plt.show()


# In[59]:

"""
lda = LatentDirichletAllocation(n_components = 25).fit(term_doc_matrix)
lda_dtf = lda.fit_transform(term_doc_matrix)
sorting = np.argsort(lda.components_)[:,::-1]
print(sorting)
features = np.array(vect.get_feature_names())
import mglearn
mglearn.tools.print_topics(topics = range(25), feature_names=features,sorting=sorting, topics_per_chunk=5,n_words=10)
doc_probs= lda.transform(term_doc_matrix)
print(doc_probs.argmax(axis=1))
"""




"""Function that creates the topics_env and topics_non_env matrices

Methodology :
1. go through all csv files that contain the case_numbers (scraped per topic from Juridict)
2. all 'environmental cases' will be in 'Aménagement de la nature, de l'environnement et urbanisme'. Hence, if 'nature' is not in the topic name, we directly save the cases in the topics_non_env matrix
3. if 'nature' is in, we save all cases as environmental except from those who are in 'non_env_to_substract' list of keywords. These keywords have been manually double-checked by matthias as being non-environmental.
"""
def merge_sub_db():

    non_env_to_substract = ['Trafic aérien',
                            'Camping',
                            'mines',
                            'Chasse/Généralités',
                            'Chasse/Loi du 28 février 1882',
                            'Chasse/Permis de chasse',
                            'Code de développement territorial',
                            'Déchets',
                            "eau navigables",
                            "eau non navigables/",
                            'Eaux/Dispositions relatives à la lutte contre les inondations',
                            "Eaux/Distribution",
                            "Eaux/Organismes",
                            'Eaux/Subventions',
                            'Logement',
                            'Monuments et sites',
                            'Remembrement',
                            'Urbanisme et aménagement du territoire',
                            'Voirie']

    #merge all sub-databases
    topics_non_env=pd.DataFrame()
    topics_env = pd.DataFrame()
    db_dir = '/users/marion1meyers/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/all_topics_csv/'
    for topic_db in os.listdir(db_dir):
        print(topic_db)

        if 'nature' not in topic_db and '.csv' in topic_db:
            print('non environmental')
            db = pd.read_csv(str(db_dir) + str(topic_db))
            del db['Unnamed: 0']
            db = db.loc[(db != 0).any(1)]
            topics_non_env = topics_non_env.append(db)
            print(topics_non_env.shape)

        elif 'nature' in topic_db and '.csv' in topic_db:
            bool = True

            for to_substract in non_env_to_substract:
                #print('checking : '+str(to_substract))
                #this is a non environmental topic
                if to_substract in topic_db:
                    print('nature but non environmental topic')
                    db = pd.read_csv(str(db_dir) + str(topic_db))
                    del db['Unnamed: 0']
                    db = db.loc[(db != 0).any(1)]
                    topics_non_env = topics_non_env.append(db)
                    print(topics_non_env.shape)
                    bool = False
            #if the name of the df was not in the 'to be excluded'
            #we still need to check if any of the columns are part of the 'to be excluded' list
            if bool:
                db = pd.read_csv(str(db_dir) + str(topic_db))
                del db['Unnamed: 0']
                db = db.loc[(db != 0).any(1)]

                for col in db.columns:
                    #print(col)
                    for to_substract in non_env_to_substract:
                        #print('checking : ' + str(to_substract))
                        #this is a column that is should be non-environmental
                        if to_substract in col:
                            #print('column : '+str(col))
                            print('non environmental column')
                            new_df = db[col]
                            del db[col]
                            topics_non_env = topics_non_env.append(new_df)
                            print(topics_non_env.shape)
                            bool2 = False
                print('environmental minus non-environmental columns')
                topics_env = topics_env.append(db)
                print(topics_env.shape)

    topics_env = topics_env.fillna(0)
    topics_non_env = topics_non_env.fillna(0)
    topics_env.to_csv('/users/marion1meyers/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/all_topics_csv/topics_env_NEW.csv')
    topics_non_env.to_csv('/users/marion1meyers/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/all_topics_csv/topics_non_env_NEW.csv')


# In[14]:


#topics_env.to_csv('C:/users/mario/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/marion1meyers/topics_env.csv')
#topics_non_env.to_csv('C:/users/mario/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/marion1meyers/topics_non_env.csv')

#topics_env = pd.read_csv('C:/users/mario/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/marion1meyers/topics_env.csv')
#topics_non_env = pd.read_csv('C:/users/mario/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/marion1meyers/topics_non_env.csv')
#del topics_non_env['Unnamed: 0']
#del topics_non_env['Unnamed: 0']


# ## Steps
# ### 1. Retrieve list of case numbers from the csv file
# ### 2. retrieve the list of term_dox matric from our archive
# ### 3. draw conclusions on the words used
"""Creates the case_term_doc_matrix_env and case_term_doc_matrix_non_env
1. retrieve all the case numbers from the given matrix (either topics_env or topics_non_env)
2. create an entry for each case withe case_number, topic, link to vocabulary file (currently null)"""
def create_case_topic_df(topics_env):
    counter=0
    cases_count=0
    term_doc_list=list()

    case_topic_termdoc = pd.DataFrame(columns=['case_number', 'topic', 'term_doc_matrix'])

    #dir = "C:/users/mario/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive/"
    #for ind, column in enumerate(topics_env.columns):
    for ind, column in enumerate(topics_env.columns):
        #print(ind, column)
        #case_numbers = topics_env[column]
        case_numbers = topics_env[column]
        case_numbers= case_numbers[case_numbers!=0]
        for case_number in case_numbers:
            #row = pd.Series('case_number':case_number,'topic': column,'term_doc_matric': 0)
            #row = pd.DataFrame([case_number, column, 0])
            case_topic_termdoc = case_topic_termdoc.append({'case_number':int(case_number), 'topic':column,'term_doc_matrix':0}, ignore_index=True)
            #case_topic_termdoc = case_topic_termdoc.append(row, ignore_index=True)
        cases_count = cases_count+len(case_numbers)


    case_topic_termdoc = case_topic_termdoc.set_index(['case_number'])
    case_topic_termdoc = case_topic_termdoc.loc[~case_topic_termdoc.index.duplicated(keep='first')]
    print('case topic shape: '+str(case_topic_termdoc.shape))
    case_topic_termdoc.to_csv('/users/marion1meyers/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/saved_dataframes/case_topic_term_env_NEW.csv')

"""This function fills the term_doc_matrix column from the case_topic_termdoc matrix. It also adds a 'text' that we realized would be useful in implementing the machine learning algorithm
1. for each year, go through all cases and retrieve their case number
2. if the case_number is also in the case_topic_termdoc, add the path to its vocabulary to the termdoc column. then, change its pdf to text, and add the full string to the 'text' column
"""
def link_juri_and_full_dataset(case_topic_termdoc):
    #add a 'text' column, this will facilitate our use of the `CountVectorizer sklearn feature that takes in a list of text
    texts = [None] * len(case_topic_termdoc)
    case_topic_termdoc['case_text'] = texts

    #case_topic_termdoc = pd.read_csv('C:/users/mario/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/marion1meyers/case_topic_term_non_env.csv')
    #go through the whole archive and whenever we find a case that is present in the dataframe, append its term_doc matrix to the one above
    #archive_dir = "/Users/marion1meyers/Documents/LawTechLab/environmental-args-belgian-law/Conseil_Etat_Archive_New/"
    archive_dir = "data/full_dataset/"
    true_counter =0
    false_counter=0
    for year in os.listdir(archive_dir):
        print(year)
        #french_dir = archive_dir+year+'/french/'
        french_dir = archive_dir+str(year)+'/french/'

        for case in os.listdir(french_dir):
            #print(case)
            if 'pdf' in case:

                if 'trad' in case:
                    case2=str(case)
                    case2 = case2[0:len(case2)-9]
                    #case_number = case[len(case)-5: len(case)]
                else:
                    case2=case[0:len(case)-4]
                parts = case2.split('_')
                case_number=parts[2]
                case_number = ''.join(c for c in case_number if c.isdigit())
                #print(case_number)
                case_number = int(case_number)
                #else:
                 #   case=str(case)
                  #  case = case[0:len(case)-4]
                   # case_number = case[len(case)-5: len(case)]
                #print(case_number)

                #add the corresponding csv file path to the case_n, topic database
                #if this is more than one row than this means that the same case can be classified in different topics
                if case_number in case_topic_termdoc.index:
                    true_counter = true_counter+1
                    #print(french_dir+str(case_number)+'.csv')
                    #case_topic_termdoc.loc[case_number]['term_doc_matrix'] = str(french_dir+str(case_number)+'.csv')
                    case_topic_termdoc.at[case_number, 'term_doc_matrix'] = french_dir+str(case_number)+'.csv'
                    try:
                        case_text = pdf_to_text(french_dir+str(case))
                    #print(case_text)
                        case_topic_termdoc.at[case_number, 'case_text'] = case_text
                    except:
                        print('not pdf')
                    #print('string length ' + str(len(case_topic_termdoc.loc[case_number]['case_text'])))
                    #print(case_topic_termdoc.loc[case_number]['term_doc_matrix'])
                else:
                    false_counter = false_counter+1
                #print(case_number in case_topic_termdoc.index)
        print('true counter: '+str(true_counter))
        print('false counter: '+str(false_counter))
            #case_topic_termdoc.loc[case_number]['term_doc_matrix'] = dir+str(case_number)+'.csv'
    #case_topic_termdoc.to_csv('C:/users/mario/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/marion1meyers/case_topic_termdoc.csv')
    return case_topic_termdoc

# In[ ]:


#case_topic_termdoc = pd.read_csv('C:/users/mario/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/marion1meyers/case_topic_term_non_env.csv')
#case_topic_termdoc = pd.read_csv('/users/marion1meyers/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/saved_dataframes/case_topic_term_non_env_NEW.csv', dtype={'term_doc_matrix':'str'})
case_topic_termdoc = pd.read_csv('data/all_topics_csv/saved_dataframes/case_topic_term_non_env_NEW.csv', dtype={'term_doc_matrix':'str'})


case_topic_termdoc = case_topic_termdoc.set_index(['case_number'])
case_topic_termdoc = case_topic_termdoc.loc[~case_topic_termdoc.index.duplicated(keep='first')]
print('case topic term shape: '+str(case_topic_termdoc.shape))
case_topic_termdoc = link_juri_and_full_dataset(case_topic_termdoc)
#case_topic_termdoc.to_csv('/users/marion1meyers/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/saved_dataframes/case_topic_term_non_env_full_NEW.csv')
case_topic_termdoc.to_csv('data/all_topics_csv/saved_dataframes/case_topic_term_non_env_full_NEW.csv')

# i should save it here
#case_topic_termdoc.to_csv('/users/mario/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/saved_dataframes/case_topic_term_non_env.csv')
#no_link = case_topic_termdoc[case_topic_termdoc.term_doc_matrix == '0']
#print('no link shape : '+str(no_link.shape))
#link = case_topic_termdoc[case_topic_termdoc.term_doc_matrix !='0' ]
#print('link shape : '+str(link.shape))

#merged_db = pd.DataFrame()
"""merged_db_non_env = pd.DataFrame()
for index, db_dir in enumerate(link.iloc[3000:6000].term_doc_matrix):
    print(index)
    try:
        db = pd.read_csv(db_dir)
        del db['Unnamed: 0']
        #merged_db = merged_db.append(db, sort=True)
        merged_db_non_env = merged_db_non_env.append(db, sort=True)
    except Exception as e:
        print(e)
        print('csv file not found')
    #print(db)"""

"""
# In[ ]:


##imagine we have the huge merged database of the full dictionary fo all cases related to the environment. 
#word_sums = merged_db.sums().sort_values(ascending=False)

#delete the words that seem to be used in all cases. words like 'case', 'law' and these kinds of stuff. 
#we want to reach the point where the words we retrieved are specific enough to  have relevant information about the cases 
#classified as 'nature'
#print(merged_db.shape)
#merged_db_non_env.to_csv("non_env_vocab.csv")

#merge_sub_db()
#topics_env = pd.read_csv('/users/marion1meyers/documents/LawTechLab/environmental-args-belgian-law/all_topics_juridict/saved_dataframes/topics_env_NEW.csv', index_col=False)
#del topics_env['Unnamed: 0']
#create_case_topic_df(topics_env)
#print('Code de développement territorial' in 'Aménagement du territoire, urbanisme, environnement et nature_Code de développement territorial (Région wallonne).csv')

"""