# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:03:53 2019

@author: Aline
"""

##############################################################################
import pandas as pd
import numpy as np
import config
import re
  

raw = pd.read_csv(config.RAW_DATA_PATH)
clean_dataset = raw.copy()
print('number or rows in the raw dataset:', clean_dataset.shape[0])

# drop duplicates based on id
clean_dataset.drop_duplicates(subset = 'id', inplace=True)
clean_dataset.reset_index(inplace=True, drop=True)
print('number of rows after dropping id duplicates:', clean_dataset.shape[0])

#drop duplicates of titles
clean_dataset.drop_duplicates(subset = 'title', inplace=True)
clean_dataset.reset_index(inplace=True, drop=True)
print('number of rows after dropping title duplicates:', clean_dataset.shape[0])

# first clean abstracts from undesired characters
clean_dataset.abstract = replace_str_abstracts(clean_dataset.abstract)

#  clean titles from undesired characters
clean_dataset.title = replace_str_titles(clean_dataset.title)

# remove comments, proceedings, and discussions
list_ids = get_list_nonabstracts(clean_dataset)
clean_dataset.drop(list_ids, inplace=True)
clean_dataset.reset_index(inplace=True, drop=True)
print(len(list_ids), 'non-abstracts removed')

# remove abstracts that are too short
list_ids = remove_short_texts(clean_dataset.abstract)
clean_dataset.drop(list_ids, inplace=True)
clean_dataset.reset_index(inplace=True, drop=True)
print(len(list_ids), 'short texts removed')

# save cleaned dataset
try:
    clean_dataset.to_csv(config.CLEAN_DATA_PATH, index=False)
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
    unhelpful items"""
    ids = []
    for i in range(data.shape[0]):
        if data.title[i].startswith('Proceedings'):
            ids.append(i)
            print('proc', data.id[i])
        if data.abstract[0].find('extended abstracts') >= 0:
            ids.append(i)
            print('extended abstracts', data.id[i])
    return ids
        
def get_text_wordcount(data):
    """returns a list with the word count of each item"""
    word_count = []
    for i in data:
        word_count.append(len(i.split()))
    return word_count
        
def remove_short_texts(data):
    """ remove items shorter than mean count minus 40% of one standard deviation"""
    ids = []
    wc = get_text_wordcount(data)
    min_keep = np.mean(wc)-(np.std(wc)*1.4)
    for i in range(len(data)):
        if len(data[i].split()) < min_keep: ids.append(i)
    return ids

def remove_url(str):
    res = re.sub(r'^https?:\/\/.*[\r\n]*', '', str, flags=re.MULTILINE)
    return res

def replace_str_abstracts(data):
    data = data.str.lower()
    data = data.str.strip()
    data = data.apply(remove_url)
    data = data.str.replace("#", " ")
    data = data.str.replace("--", ", ")
    data = data.str.replace(r"\n", " ")
    data = data.str.replace("~", " ")
    data = data.str.replace(" / ", " ")
    data = data.str.replace("/", "-")
    data = data.str.replace("|", " ")
    data = data.str.replace(": "," ")
    data = data.str.replace(";", ".")
    data = data.str.replace(r'\\"', '')
    data = data.str.replace('"', " ")
    data = data.str.replace("\%$", "% ")
    data = data.str.replace("$", "")
    data = data.str.replace(r"\\&", " and ")
    data = data.str.replace(" & ", " and ")
    data = data.str.replace(r"\\textit", " ")
    data = data.str.replace(r"\\texttt", " ")
    data = data.str.replace(r"\\url", " ")
    data = data.str.replace(r"\\alpha", "alpha ")
    data = data.str.replace(r"\\bf", " ")
    data = data.str.replace(r"\\textbf", " ")
    data = data.str.replace(r"\\textth", " ")
    data = data.str.replace(r"\\'", " ")
    data = data.str.replace("{","")
    data = data.str.replace("}","")
    data = data.str.replace(r"\\mathcal"," ")
    data = data.str.replace(r"\\cite", "")
    data = data.str.replace(r"\\infty"," ")
    data = data.str.replace(r"\\math"," ")
    data = data.str.replace(r"\\text", " ")
    data = data.str.replace(r"\\tilde"," ")
    data = data.str.replace(r"\\ell"," ")
    data = data.str.replace(r"\\sqrt"," ")
    data = data.str.replace(r"\\log_"," ")
    data = data.str.replace(r"\sim"," ")
    data = data.str.replace(r"\boldsymbol"," ")
    data = data.str.replace(r"\\em"," ")
    data = data.str.replace(r"\\it", " ")
    data = data.str.replace(r"\\^", "")
    data = data.str.replace(r"\\mbox", " ")
    data = data.str.replace(r"\\underline", " ")
    data = data.str.replace(r"\\epsilon", "epsilon")
    data = data.str.replace(r"\\cite", " ")
    data = data.str.replace("`", " ")
    data = data.str.replace(r"\\emph"," ")
    data = data.str.replace(r"\\sqrt","sqrt ")
    data = data.str.replace(r"\\log", "log ")
    
    # last
    data = data.str.replace(r"\\", "")
    data = data.str.replace("   "," ")
    data = data.str.replace("  ", " ")

    return data


def replace_str_titles(data):
    data = data.str.lower()
    data = data.str.strip()
    data = data.str.replace(r"\n", " ")
    data = data.str.replace("?", " ")
    data = data.str.replace("--", ", ")
    data = data.str.replace("~", " ")
    data = data.str.replace(" / ", " ")
    data = data.str.replace("/", " ")
    data = data.str.replace("|", " ")
    data = data.str.replace(":", ",")
    data = data.str.replace(";", ".")
    data = data.str.replace('"', " ")
    data = data.str.replace("'", "")
    data = data.str.replace("`", "")
    data = data.str.replace("Ã©", "e")
    data = data.str.replace("$", "")
    data = data.str.replace(" & ", " and ")
    data = data.str.replace(r"\\'", " ")
    data = data.str.replace("{", "")
    data = data.str.replace("}", "")
    data = data.str.replace(r"\\^", "")
    data = data.str.replace("`", " ")
    # last
    data = data.str.replace("   ", " ")
    data = data.str.replace("  ", " ")

    return data