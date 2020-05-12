#!/usr/bin/env python
#!/usr/bin/env python
# coding: utf-8

# In[55]:


#Import packages

import pandas as pd
from selenium import webdriver
import time
import numpy as np
#from pyvirtualdisplay import Display
import datetime

"""This script was used to scraped all classified topics from the websiote Juridict. """

def get_topics(topic_name, topics_df, url_add_on, topic_column_id):
    #chromeBrowser = webdriver.Chrome(executable_path='/usr/bin/chromedriver')
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--window-size=1420,1080')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chromeBrowser = webdriver.Chrome(chrome_options=chrome_options)
    basic_url = 'http://juridict.conseildetat.be/'
    url = str(basic_url)+str(url_add_on)
    #print('get url : '+str(url))
    #print(topic_column_id)
    #print('type : '+str(type(topic_column_id)))
    #print('topic column id : '+str(topic_column_id))
    #chromeBrowser = webdriver.Chrome(executable_path='C:/Program Files/chromedriver.exe') 
    chromeBrowser.get(url)       # 1 
    time.sleep(2)                              
    
    #check if this is a leaf node : meaning we retrieve the file content instead of 
    if check_if_end_tree(chromeBrowser):
        print('found end, this is where you would save the cases id in the dataframe')
        cases_id_list = get_cases(url)
        full_row = np.zeros(500)
        for index,case_id in enumerate(cases_id_list):
            full_row[index] = case_id
        topics_df[topic_name] = full_row
        chromeBrowser.close()
        return False
    
    else:
        #get the left side column
        lef_column = chromeBrowser.find_element_by_id(topic_column_id)
        topics = lef_column.find_elements_by_tag_name('a') 
        topics_id =list()
        topics_id_for_url=list()
        topics_names=list()
        topic_names = [topic.text for topic in topics]
        topic_ids = [topic.get_attribute("id") for topic in topics]
        topic_style=[topic.get_attribute("style") for topic in topics]
        chromeBrowser.close()
        for index, new_topic_name in enumerate(topic_names):
            #print(topic.text)

            new_topic_name = topic_name+'/'+new_topic_name
            print('new topic name : '+str(new_topic_name))
            topic_id = topic_ids[index]
            #print('element style : '+str(topic.get_attribute("style")))
            if topic_style[index] != 'color: rgb(153, 153, 153);':
                topics_id.append(topic_id)

                topic_id_for_url = topic_id[1:len(topic_id)]
                topic_id_for_url = topic_id_for_url.replace('_',':')
                topics_id_for_url.append(topic_id_for_url)
                #first check if the topic is empty
                try:
                	get_topics(new_topic_name, topics_df,'#arb'+str(topic_id_for_url), topic_id[1:len(topic_id)])
                except:
                    print('ERROR : we save the file')
                    new_topic_name_string = new_topic_name.replace('/', '_')
                    topics_df.to_csv('/home/marion1meyers/'+new_topic_name_string+str(datetime.datetime.now())+'.csv')
           # else:
            #    print('style detection : empty content')                                  
        #chromeBrowser.close()
        return True


def check_if_end_tree(chromeBrowser):
    content = chromeBrowser.find_element_by_id('arr_tbody')
    trs = content.find_elements_by_tag_name('tr')
    print('content length : '+str(len(trs)))
    if len(trs) > 0:
        return True
    else:
        return False
    

def get_cases(url):
    #chromeBrowser = webdriver.Chrome(executable_path='/usr/bin/chromedriver')
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--window-size=1420,1080')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chromeBrowser = webdriver.Chrome(chrome_options=chrome_options)

    #chromeBrowser = webdriver.Chrome()
    chromeBrowser.get(url)       # 1 
    time.sleep(2)                              
    content = chromeBrowser.find_element_by_id('arr_tbody')
    trs = content.find_elements_by_tag_name('tr')
    
    list_cases = list()
    for tr in trs:
        #print(tr.text)
        heads = tr.find_elements_by_class_name('td_data')
        if len(heads)>0:
            case_number = heads[0]
            #print('case number : '+str(case_number.text))
            list_cases.append(case_number.text)
    list_cases = list(dict.fromkeys(list_cases))
    chromeBrowser.close()
    return list_cases


"""topics_df = pd.DataFrame(index=range(0,500))
topic_name =''
#chromeBrowser = webdriver.Chrome(executable_path='/usr/bin/chromedriver')
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--window-size=1420,1080')
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chromeBrowser = webdriver.Chrome(chrome_options=chrome_options)
get_topics(topic_name, topics_df, chromeBrowser, '#arbJ:0:225:195','J_0_225_195')

topics_df.to_csv('/home/marion1meyers/topics_df_urb_bxl_new.csv')"""


#let's get it more separate otherwise it throws errors
topic_names=['Generalites', 'Affaires sociales et Santé Publique', 'Aménagement du territoire, urbanisme, environnement et nature', "Conseil d'Etat et juridictions administratives", "Contrats de l'administration", "Economie", "Enseignement, culture, juenesse et sport", "Etrangers","Fonction publique","Intérieur","Justice","Pouvoirs subordonnés","Professions","Varia"]
topics = list(range(223,237))

for index, topic_index in enumerate(topics):
    topic_name = topic_names[index]
    print(topic_name)
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--window-size=1420,1080')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chromeBrowser = webdriver.Chrome(chrome_options=chrome_options)

    #get the list of subtopics for this specific topic
    basic_url = 'http://juridict.conseildetat.be/'
    url = str(basic_url) + "#arbJ:0:"+str(topic_index)
    #print(url)
    chromeBrowser.get(url)  # 1
    time.sleep(2)
    # get the left side column
    topic_column_id = "J_0_"+str(topic_index)
    lef_column = chromeBrowser.find_element_by_id(topic_column_id)
    topics = lef_column.find_elements_by_tag_name('a')
    topics_id = list()
    topics_id_for_url = list()
    topics_names = list()
    for topic in topics:
        print(topic.text)
        new_topic_name = topic_name + '/' + topic.text
        new_topic_name_string = new_topic_name.replace('/','_')
        print('new topic name : ' + str(new_topic_name))
        topic_id = topic.get_attribute("id")
        topic_id_for_url = topic_id[1:len(topic_id)]
        topic_id_for_url = topic_id_for_url.replace('_', ':')
        topics_id_for_url.append(topic_id_for_url)
        topics_df = pd.DataFrame(index=range(0, 500))
        get_topics(new_topic_name, topics_df,'#arb'+str(topic_id_for_url), topic_id[1:len(topic_id)])
        topics_df.to_csv('/home/marion1meyers/'+new_topic_name_string+'.csv')
        #chromeBrowser.close()
    chromeBrowser.close()





