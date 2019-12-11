from flask import Flask, jsonify, request, render_template

import joblib
import pandas as pd
import numpy as np
from gensim.models import doc2vec
from gensim.models.ldamodel import LdaModel
import os

app = Flask(__name__, static_url_path='', static_folder='templates')

data = pd.read_csv("data/clean_dataset.csv")

# load pre-saved doc2vec model
model_d2v = doc2vec.Doc2Vec.load('models/doc2vec_all_abstracts.model')
# load corpus doc2vec
corpus_d2v = joblib.load('corpus/all_abstracts_for_doc2vec.corpus')
# Load LDA based on the titles
lda = LdaModel.load('models/LDA_titles_30topics')

@app.route('/', methods=['GET', 'POST'])
def index():
    ids = np.random.randint(0, len(data), 3)
    abstracts = data.abstract[ids]
    titles = data.title[ids]
    arxiv_id = data.id[ids]
    abstracts.reset_index(drop=True, inplace=True)
    titles.reset_index(drop=True, inplace=True)
    arxiv_id.reset_index(drop=True, inplace=True)
    return render_template('page1.html',
                           abstract1=abstracts[0], abstract2=abstracts[1], abstract3=abstracts[2],
                           title1=titles[0], title2=titles[1], title3=titles[2],
                           id1 = arxiv_id[0], id2 = arxiv_id[1], id3 = arxiv_id[2]) #**locals()


@app.route('/choose', methods=['GET', 'POST'])
def load_similar():
    arxivid = request.form['chosen_id']
    target = arxivid_to_numid(arxivid)
    list_sim = get_top_similar_word2vec(target)
    result_slice = data[data.id.isin(list_sim)]
    result_slice.reset_index(drop=True, inplace=True)
    return render_template('index.html',
                           abstract1=data.abstract[target], title1=data.title[target],  id1 = arxivid,
                           sug_abstract1=result_slice.abstract[1],
                           sug_abstract2=result_slice.abstract[2],
                           sug_abstract3=result_slice.abstract[3],
                           sug_abstract4=result_slice.abstract[4],
                           sug_abstract5=result_slice.abstract[5])

def get_top_similar_word2vec(num_id):
    list_sim = []
    inferred_vector = model_d2v.infer_vector(corpus_d2v[num_id].words)
    sims = model_d2v.docvecs.most_similar([inferred_vector], topn=6)
    for abs in sims:
        list_sim.append(numid_to_arxivid(abs[0]))
    return list_sim

def numid_to_arxivid(num_id):
    return data.id[num_id]

def arxivid_to_numid(arxiv_id):
    return data[data.id == arxiv_id].index[0]


if __name__ == "__main__":
    app.run(debug=True)
