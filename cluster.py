# -*- coding: utf-8 -*-
"""
@author: Aline
"""
import spacy
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

def create_dict_texts(data, file):
    """ creates a dictionary of texts and respective vectors, returns two lists,
    one with the keys and another with values"""
    sent_vectors = {}
    docs = []
    for item in data:
        doc = nlp(item)
        docs.append(doc)
        sent_vectors.update({item: doc.vector})
    texts = list(sent_vectors.keys())
    vectors = list(sent_vectors.values())
    return texts, vectors

def get_epsilons_DBSCAN(vectors):
    n_classes = {}
    for epsi in np.arange(0.001, 1, 0.002):
        temp_dbscan = DBSCAN(eps=epsi, min_samples=2, metric='cosine').fit(vectors)
        n_classes.update({epsi: len(pd.Series(temp_dbscan.labels_).value_counts())})
    plt.plot(list(n_classes.keys()), list(n_classes.values()))
    return n_classes

def get_groups_DBSCAN(epsilon, vectors):
    groups_dbscan = DBSCAN(eps=epsilon, min_samples=2, metric='cosine').fit(vectors)
    return groups_dbscan

####################################################
if __name__ == '__main__':
    # load the data
    clean_dataset = pd.read_csv('data/clean_dataset.csv')

    list_abs, vector_abs = create_dict_texts(clean_dataset.abstract[:30], 'processed/abstract_vectors3000.csv')
    list_titles, vector_titles = create_dict_texts(clean_dataset.title[:3000], 'processed/title_vectors3000.csv')

    x = np.array(vector_abs)
    epsis = get_epsilons_DBSCAN(x)

    final_epsilon = 0.05
    groups_dbscan = get_groups_DBSCAN(final_epsilon, x)

    dbscan_dataset = pd.DataFrame({'label': groups_dbscan.labels_, 'abstract': list_abs})
    filt = dbscan_dataset[dbscan_dataset.label == 0].abstract.to_list()
