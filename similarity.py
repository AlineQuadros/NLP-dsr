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
import joblib
##############################################################################

# load the USE - universal-sentence-encoder from Google
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
# load the data
clean_dataset = pd.read_csv('data/clean_dataset.csv')
corpus_tokenized = joblib.load('corpus/all_abstracts_tokenized.corpus')
kmeans_dataset = pd.read_csv('processed/abstracts_kmeans_40clusters.csv')

# calculate pairwise similarity within clusters
for i in set(kmeans_dataset.k_means_cluster):
    ids_arxiv = kmeans_dataset.id[kmeans_dataset.k_means_cluster == i]
    ids_num = kmeans_dataset.id[kmeans_dataset.k_means_cluster == i].index.to_list()
    assert len(ids_arxiv) == len(ids_num)
    corpus_slice = [corpus_tokenized[i] for i in ids_num]
    calculate_cluster_USE(corpus_slice, ids_arxiv, i)

##############################################################################

def calculate_cluster_USE(corpus, ids_arxiv, cluster_id):
    for i in range(len(corpus)):
        # calculate similarity ALL vs. ALL within each cluster
        print(i)
        similarity_USE = pd.DataFrame()
        ab1 = corpus[i]
        e1 = embed(ab1)["outputs"]
        #for j in np.arange(i+1, len(corpus)):
        for j in range(len(corpus)):
            print(j)
            ab2 = corpus[j]
            e2 = embed(ab2)["outputs"]
            sim = get_similarity_pairs_USE(e1, e2)
            similarity_USE = similarity_USE.append(pd.Series([ids_arxiv.iloc[i], ids_arxiv.iloc[j], round(sim, 3)]), ignore_index=True)
        similarity_USE.columns = ['ab1', 'ab2', 'sim_USE']
        similarity_USE.to_csv('similarity/similarity_USE_cluster'+str(cluster_id)+'.csv', index=False)


def calculate_overall_USE(data):
    for i in range(len(data)):
        # calculate similarity ALL vs. ALL
        print(i)
        list_similarity_USE = []
        list_sim = []
        ab1 = tokenize.sent_tokenize(data.abstract[i])
        e1 = embed(ab1)["outputs"]
        for j in np.arange(i + 1, len(data)):
            ab2 = tokenize.sent_tokenize(data.abstract[j])
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

def assemble_dataset_from_files():
    dataset_USE = pd.DataFrame()
    for f in glob.glob('similarity/*.csv'):
        print('reading data from file {}...'.format(f))
        txt = open(f, 'r')
        for l in txt:
            l = l.replace("\'", "\"")
            line = json.loads(l)
            dataset_USE = dataset_USE.append(pd.Series([line['ab1'], line['ab2'], line['sim_use']]), ignore_index=True)
        #dataset_USE = dataset_USE.append(raw, ignore_index=True)
    dataset_USE.columns = ['ab1', 'ab2', 'sim_use']
    dataset_USE.to_csv("data/similarity_USE.csv", index=False)

##################################################################################################################
