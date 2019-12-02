# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:46:19 2019

@author: Aline

Functions to calculate similarity between abstracts

"""
##############################################################################
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from nltk import tokenize
import glob
import json
import csv
import config

##############################################################################

def calculate_overall_USE(data):
    for i in range(len(data)):
        # calculate similarity ALL vs. ALL
        print(i)
        list_similarity_USE = []
        list_sim = []
        for j in np.arange(i + 1, len(data)):
            ab1 = tokenize.sent_tokenize(data.abstract[i])
            ab2 = tokenize.sent_tokenize(data.abstract[j])
            e1 = embed(ab1)["outputs"]
            e2 = embed(ab2)["outputs"]
            sim = get_similarity_pairs_USE(e1, e2)
            line = "{'ab1':'" + str(data.id[i]) + "','ab2':'" + str(data.id[j]) + "','sim_use':" + str(round(sim, 3)) + "}"
            list_sim.append(line)
        temp = 'similarity/iteration' + str(i) + '_' + str(j) + '.csv'
        with open(temp, 'w+') as f:
            f.write('\n'.join(list_sim))
            f.close()


def get_similarity_pairs_USE(e1, e2):
    """ calculates the mean of the maximum similarity between one sentence of one abstract against
     all sentences of the other abstract"""
    # iterate over the sentences of the first abstract
    list_sims = []
    for s in range(e1.shape[0]):
        max_sim = 0
        # iterate over the sentences of the second abstract
        for s2 in range(e2.shape[0]):
            # calculate similarity
            sim = np.inner(e1[s], e2[s2])
            if sim > max_sim: max_sim = sim
        list_sims.append(max_sim)
    for s in range(e2.shape[0]):
        max_sim = 0
        # iterate over the sentences of the second abstract
        for s2 in range(e1.shape[0]):
            # calculate similarity
            sim = np.inner(e2[s], e1[s2])
            if sim > max_sim: max_sim = sim
        list_sims.append(max_sim)

    return sum(list_sims)/(len(list_sims))

#################################################################################

def assemble_dataset_from_files():
    dataset_USE = pd.DataFrame()
    for f in glob.glob("similarity/*.csv"):
        txt = open(f, "r")
        for l in txt:
            l = l.replace("\'", "\"")
            line = json.loads(l)
            dataset_USE = dataset_USE.append(pd.Series([line['ab1'], line['ab2'], line['sim_use']]), ignore_index=True)
        #dataset_USE = dataset_USE.append(raw, ignore_index=True)

if __name__ == '__main__':
    # load the USE - universal-sentence-encoder from Google
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
    # load the data
    clean_dataset = pd.read_csv('data/clean_dataset.csv')
    #clean_dataset = pd.read_csv('clean_dataset.csv')
    # calculate pairwise similarity and dump to json files
    calculate_overall_USE(clean_dataset)
    # load json data from multiple files into unique dataframe
    assemble_dataset_from_files()

