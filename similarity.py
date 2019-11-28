# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:46:19 2019

@author: Aline

Functions to calculate similarity between abstracts

"""
##############################################################################

import numpy as np
import pandas as pd
import spacy
import sys
import gensim

import tensorflow_hub as hub
from nltk import tokenize
import numpy as np
import config

# load the USE - universal-sentence-encoder from Google
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
# USAGE of USE:
#embeddings = embed([
#    "the crabs live in mangroves in india",
#    "the crabs live in mangroves"])["outputs"]
#corr = np.inner(embeddings[0], embeddings[1])
#print(corr)
#print(embeddings)
calculate_overall_USE(clean_dataset.abstract)

##############################################################################
def calculate_overall_USE(data):
    for i in range(len(data)):
        #calculate similarity ALL vs. ALL
        print(i)
        list_similarity_USE = []
        for j in range(len(data)):
            ab1 = tokenize.sent_tokenize(data[i])
            ab2 = tokenize.sent_tokenize(data[j])
            e1 = embed(ab1)["outputs"]
            e2 = embed(ab2)["outputs"]
            list_similarity_USE.append([i, j, get_similarity_pairs_USE(e1, e2)])
        # update the file at every completed j
        list_similarity_USE = pd.DataFrame(list_similarity_USE)
        try:
            list_similarity_USE.to_csv(config.SIMILARITY_USE_PATH, mode='a', index=False, header=False)
        except:
            print('error at i:', i,' and j:', j)


def get_similarity_pairs_USE(e1, e2):
    # iterate over the sentences of the first abstract
    list_sims = []
    for s in range(e1.shape[0]):
        max_sim = 0
        # iterate over the sentences of the second abstract
        for s2 in range(e2.shape[0]):
            sim = np.inner(e1[s], e2[s2])
            if sim > max_sim: max_sim = sim
        list_sims.append(max_sim)
    
    return sum(list_sims)/(e1.shape[0])   

