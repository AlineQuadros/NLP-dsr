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
import os
import en_core_web_lg
from sklearn.cluster import KMeans

nlp = en_core_web_lg.load()

# nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load('en_core_web_lg')

def get_abstract_vectors(data, filevectors, fileids):
    """ gets a vector representation of the abstracts based on a language model
        creates a dictionary of texts and respective vectors, returns two lists,
        one with the ids and another with vectors"""
    sent_vectors = {}
    docs = []
    for i in range(len(data)):
        doc = nlp(data.abstract[i])
        sent_vectors.update({data.id[i]: doc.vector})
    abstract_ids = list(sent_vectors.keys())
    abstract_vectors = list(sent_vectors.values())
    # dump vectors to npy file
    x = np.array(abstract_vectors)
    np.save(filevectors, x, allow_pickle=False)
    # save ids to text file
    with open(fileids, 'w+') as f:
        f.writelines("%s\n" % item for item in abstract_ids)
    return abstract_ids, abstract_vectors

def get_epsilons_DBSCAN(vectors):
    n_classes = {}
    for epsi in np.arange(0.001, 0.05, 0.001):
        print('------------ epsilon', epsi)
        temp_dbscan = DBSCAN(eps=epsi, min_samples=5, metric='cosine').fit(vectors)
        n_classes.update({epsi: len(pd.Series(temp_dbscan.labels_).value_counts())})
        print('number unclassified', (temp_dbscan.labels_ == -1).sum())
        print('number groups', len(pd.Series(temp_dbscan.labels_).value_counts()))
        print(pd.Series(temp_dbscan.labels_).value_counts().head(10))
    plt.plot(list(n_classes.keys()), list(n_classes.values()))
    return n_classes

def get_groups_DBSCAN(epsilon, vectors):
    groups_dbscan = DBSCAN(eps=epsilon, min_samples=2, metric='cosine').fit(vectors)
    return groups_dbscan


def kmeans_cluster(vectors, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    print(pd.Series(km.labels_).value_counts())
    return km


####################################################################################################################

if __name__ == '__main__':
    # load the data
    clean_dataset = pd.read_csv('data/clean_dataset.csv')

    # load or create file with vectors of each abstract
    if os.path.exists('processed/abstract_vectors5000.npy'):
        abstract_vectors = np.load('processed/abstract_vectors5000.npy')
        abstract_ids = pd.read_csv('processed/abstract_vectors5000_id.csv')
    else:
        abstract_ids, abstract_vectors = get_abstract_vectors(clean_dataset.loc[:5000, ['id', 'abstract']],
                                                           'processed/abstract_vectors5000.npy',
                                                           'processed/abstract_vectors5000_id.csv')

    ###### K-means
    km = kmeans_cluster(abstract_vectors, 15)
    kmeans_dataset = pd.DataFrame({'label': km.labels_, 'abstract': abstract_ids})
    kmeans_dataset.to_csv('processed/kmeans_clusters.csv', index=False)

    ####### DBSCAN
    x = np.array(abstract_vectors)
    # find the optimal number of clusters
    epsis = get_epsilons_DBSCAN(x)
    final_epsilon = 0.023
    groups_dbscan = get_groups_DBSCAN(final_epsilon, x)
    dbscan_dataset = pd.DataFrame({'label': groups_dbscan.labels_, 'abstract': abstract_ids})
    dbscan_dataset.to_csv('processed/dbscan_clusters.csv', index=False)
    print(dbscan_dataset['label'].value_counts().head(3))
    filt = dbscan_dataset[dbscan_dataset.label == 0].abstract.to_list()







def missing_elements(li):
    start, end = li[0], li[-1]
    return sorted(set(range(start, end + 1)).difference(li))


txt1 = 'we conduct an empirical study of machine learning functionalities provided by major cloud service providers, which we call machine learning clouds. machine learning clouds hold the promise of hiding all the sophistication of running large-scale machine learning instead of specifying how to run a machine learning task, users only specify what machine learning task to run and the cloud figures out the rest. raising the level of abstraction, however, rarely comes free - a performance penalty is possible. how good, then, are current machine learning clouds on real-world machine learning workloads? we study this question with a focus on binary classication problems. we present mlbench, a novel benchmark constructed by harvesting datasets from kaggle competitions. we then compare the performance of the top winning code available from kaggle with that of running machine learning clouds from both azure and amazon on mlbench. our comparative study reveals the strength and weakness of existing machine learning clouds and points out potential future directions for provement.'

txt2 = 'we conduct an study of machine learning provided by major component service providers, which we call machine learning clouds. machine learning clouds hold the promise of hiding all the sophistication of running big machine learning instead of specifying how to run a machine learning task, users only specify what machine learning task to run and the cloud figures out the rest. raising the level of abstraction, however, rarely comes free - a performance penalty is possible. how good, then, are current machine learning clouds on real-world machine learning workloads? we study this question with a focus on binary classication problems. we present mlbench, a novel benchmark constructed by harvesting datasets from kaggle competitions. we then compare the performance of the top winning code available from kaggle with that of running machine learning clouds from both azure and amazon on mlbench. our comparative study reveals the strength and weakness of existing machine learning clouds and points out potential future directions for provement.'
