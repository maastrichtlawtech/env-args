from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

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



