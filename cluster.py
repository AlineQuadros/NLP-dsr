# -*- coding: utf-8 -*-
"""
@author: Aline
"""
import spacy
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import en_core_web_lg
from sklearn.cluster import KMeans
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import pairwise_distances
from hdbscan import HDBSCAN

nlp = en_core_web_lg.load()
###################################################################################################################

# load the data
clean_dataset = pd.read_csv('data/clean_dataset.csv')
# load saved doc2vec embeddings
doc2vec_model = Doc2Vec.load('models/doc2vec_all_abstracts.model')

###### K-means
cluster_sizes = [5, 30, 40, 50, 60, 70, 80, 100, 120]
kmeans_elbow_method(cluster_sizes, doc2vec_model.docvecs.doctag_syn0)

optimal_n = 40

km = kmeans_cluster(doc2vec_model.docvecs.doctag_syn0, optimal_n)
kmeans_dataset = pd.DataFrame({'id': clean_dataset.id, 'k_means_cluster': km.labels_ })
kmeans_dataset.to_csv('processed/abstracts_kmeans_'+str(optimal_n)+'clusters.csv', index=False)

##### HDBSCAN - did not work
X = np.asarray(doc2vec_model.docvecs.doctag_syn0)
distances = pairwise_distances(X, metric='cosine').astype('float64')
clusterer = HDBSCAN(algorithm='best', approx_min_span_tree=True,
                    gen_min_span_tree=True, leaf_size=40, metric="precomputed",
                    min_cluster_size=10, min_samples=None, p=None, core_dist_n_jobs=-1)
clusterer.fit(distances)
set(clusterer.labels_)

####### DBSCAN - did not work
x = np.array(doc2vec_model.docvecs.doctag_syn0)
# find the optimal number of clusters
epsis = get_epsilons_DBSCAN(x)
final_epsilon = 0.023
groups_dbscan = get_groups_DBSCAN(final_epsilon, x)
dbscan_dataset = pd.DataFrame({'label': groups_dbscan.labels_, 'abstract': abstract_ids})
dbscan_dataset.to_csv('processed/dbscan_clusters.csv', index=False)
print(dbscan_dataset['label'].value_counts().head(3))
filt = dbscan_dataset[dbscan_dataset.label == 0].abstract.to_list()

########################################################################################################

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

def kmeans_elbow_method(cluster_sizes, vectors):
    """ calculate centroid distances to cluster centroid for different numbers of clusters"""
    distortions = []
    for k in cluster_sizes:
        kmeanModel = KMeans(n_clusters=k, random_state=0).fit(vectors)
        distortions.append(sum(np.min(cdist(vectors, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / vectors.shape[0])
    # Plot the elbow
    plt.plot(cluster_sizes, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def missing_elements(li):
    start, end = li[0], li[-1]
    return sorted(set(range(start, end + 1)).difference(li))

