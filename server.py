from flask import Flask, jsonify, request, render_template

import joblib
import pandas as pd
import numpy as np
from gensim.models import doc2vec
from gensim.models.ldamodel import LdaModel
import os

app = Flask(__name__, static_url_path='', static_folder='templates')

data = pd.read_csv('data/clean_dataset.csv')
clusters = pd.read_csv('processed/abstracts_kmeans_40clusters.csv')

# load pre-saved doc2vec model
model_d2v = doc2vec.Doc2Vec.load('models/doc2vec_all_abstracts.model')
# load corpus doc2vec
#corpus_d2v = joblib.load('corpus/all_abstracts_for_doc2vec.corpus')
# Load LDA based on the titles
#lda = LdaModel.load('models/LDA_titles_30topics.model')

@app.route('/', methods=['GET', 'POST'])
def index():
    ids = np.random.randint(0, len(data), 3)
    abstracts = data.abstract[ids]
    titles = data.title[ids]
    arxiv_id = data.id[ids]
    abstracts.reset_index(drop=True, inplace=True)
    titles.reset_index(drop=True, inplace=True)
    arxiv_id.reset_index(drop=True, inplace=True)
    return render_template('index.html',
                           abstract1=abstracts[0], abstract2=abstracts[1], abstract3=abstracts[2],
                           title1=titles[0], title2=titles[1], title3=titles[2],
                           id1 = arxiv_id[0], id2 = arxiv_id[1], id3 = arxiv_id[2]) # **locals()


@app.route('/results', methods=['GET', 'POST'])
def load_similar():
    arxivid = request.form['button']
    cluster = get_cluster(arxivid)
    target = arxivid_to_numid(arxivid)
    list_sim = get_top_similar_doc2vec(target)
    result_slice = data[data.id.isin(list_sim)]
    result_slice.reset_index(drop=True, inplace=True)
    return render_template('results.html',
                           results=result_slice,
                           abstract1=data.abstract[target],
                           title1=data.title[target],
                           id1 = arxivid,
                           cluster = cluster)

@app.route('/cluster', methods=['GET', 'POST'])
def load_cluster():
    cluster_id = request.form['button']
    return render_template('cluster.html')


def get_top_similar_doc2vec(num_id):
    list_sim = []
    # inferred_vector = model_d2v.infer_vector(corpus_d2v[num_id].words)
    inferred_vector = model_d2v.docvecs[num_id]
    sims = model_d2v.docvecs.most_similar([inferred_vector], topn=6)
    for abs in sims:
        list_sim.append(numid_to_arxivid(abs[0]))
    return list_sim

def numid_to_arxivid(num_id):
    return data.id[num_id]

def arxivid_to_numid(arxiv_id):
    return data[data.id == arxiv_id].index[0]

def get_cluster(arxiv_id):
    return clusters.k_means_cluster[clusters.id == arxiv_id].astype('int32')

if __name__ == "__main__":
    app.run(debug=True)
