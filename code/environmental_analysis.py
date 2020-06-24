from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

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
        non_zeros_sums = term_doc_matrix.shape[0] - zero_read_sums
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
        path = "Conseil_Etat_Archive/"+str(year)+"/french/"
        cases = []
        french_stopwords = get_stopswords()
        #dutch_stopwords = get_stopswords_dutch()
        #r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                try:
                    #case = pdf_to_text(os.path.join(r, file))
                    case = pdftotext2(os.path.join(r, file))
                    nltk_text = get_nltk_text(case)
                    words = filter_stopwords(nltk_text,french_stopwords)
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
        path = "Conseil_Etat_Archive_New/"+str(year)+"/french/"
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
                        #case = pdf_to_text(os.path.join(r, file))
                        case = pdftotext2(os.path.join(r, file))
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






