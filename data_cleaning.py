# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:03:53 2019

@author: Aline
"""

##############################################################################
import pandas as pd
import numpy as np
import config
  

raw = pd.read_csv(config.RAW_DATA_PATH)
clean_dataset = raw.copy()

# first clean abstracts from undesired character
clean_dataset.abstract = replace_str(raw.abstract)

# remove comments, proceedings, and discussions

# remove abstracts that are too short
list_ids = remove_short_texts(clean_dataset.abstract)
clean_dataset.drop(list_ids, inplace=True)

# reset index before saving
clean_dataset.reset_index(inplace=True, drop=True)
# save cleaned dataset
try:
    clean_dataset.to_csv(config.CLEAN_DATA_PATH)
    print('clean dataset saved successfully at', config.CLEAN_DATA_PATH)
except:
    print('not possible to save')
##############################################################################

# remove proceedings
# remove starts with Comment
# remove starts with Discussion
# remove contains "extended abstracts"
def get_list_nonabstracts(data):
    """ returns the ids of abstracts from conference proceedings or other 
    unhelpfull items"""
    starts_with = ['Discussion', 'Comment']
    contains = ['extended abstracts']
    ids = []
    for i in range(data.shape[0]):
        if data.abstract[i]:
        if data.abstract[i]:
                list_ids.append(data.id)
            pass
        if data.abstract[0].find('extended abstracts') >= 0:
            ids.append(data.id)
            pass
        
    return ids
        
def get_text_wordcount(data):
    """returns a list with the word count of each item"""
    word_count = []
    for i in data:
        word_count.append(len(i.split()))
    return word_count
        
def remove_short_texts(data):
    """ remove items shorter than mean count minus one std"""
    ids = []
    wc = get_text_wordcount(data)
    min_keep = np.mean(wc)-(np.std(wc)*1.4)
    for i in range(len(data)):
        if len(data[i].split()) < min_keep: ids.append(i)
    return ids

def replace_str(data):
    data = data.str.lower()
    data = data.str.strip()
    data = data.str.replace("$","")
    data = data.str.replace("\n"," ")
    data = data.str.replace("~"," ")
    data = data.str.replace(":"," ") #not sure
    data = data.str.replace('"'," ")
    data = data.str.replace("\%$","% ")
    data = data.str.replace("."," ")
    data = data.str.replace(" & "," and ")
    data = data.str.replace("\\textit"," ")
    data = data.str.replace("{","")
    data = data.str.replace("}","")
    data = data.str.replace(r"\\mathcal"," ")
    data = data.str.replace(r"\\infty"," ")
    data = data.str.replace(r"\\mathbb"," ")
    data = data.str.replace("\\tilde"," ")
    data = data.str.replace(r"\\ell"," ")
    data = data.str.replace(r"\sqrt"," ")
    data = data.str.replace(r"\\log_"," ")
    data = data.str.replace(r"\sim"," ")
    data = data.str.replace(r"\boldsymbol"," ")
    data = data.str.replace(r"\\em"," ")
    data = data.str.replace(r"\\emph"," ")
    data = data.str.replace(r"\sqrt"," ")
    
    # last
    data = data.str.replace("  "," ")
    data = data.str.replace("   "," ")
    
    return data