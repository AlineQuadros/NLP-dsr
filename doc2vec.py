# -*- coding: utf-8 -*-
# part of the code from: https://radimrehurek.com/gensim/auto_examples/tutorials/
# run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py

import pandas as pd
import gensim
import multiprocessing
import joblib
from scipy.spatial import distance

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1
##################################################################################################

# load data
clean_dataset = pd.read_csv('data/clean_dataset.csv')
train_corpus = list(read_corpus(clean_dataset.abstract))
kmeans_dataset = pd.read_csv('processed/abstracts_kmeans_40clusters.csv')
# test_corpus = list(read_corpus(clean_dataset.abstract[41000:41591], tokens_only=True))

# save pre-processed corpus
joblib.dump(train_corpus, 'corpus/all_abstracts_for_doc2vec.corpus')

# train and save doc2vec model as file
doc2vec_model = doc2vec_train(train_corpus)
doc2vec_model.save('models/doc2vec_all_abstracts.model')


# calculate pairwise similarity based on Doc2Vec embeddings within clusters for top 50 most similar
for i in set(kmeans_dataset.k_means_cluster):
    ids_arxiv = kmeans_dataset.id[kmeans_dataset.k_means_cluster == i]
    ids_num = kmeans_dataset.id[kmeans_dataset.k_means_cluster == i].index.to_list()
    assert len(ids_arxiv) == len(ids_num)
    corpus_slice = [corpus_tokenized[i] for i in ids_num]
    get_similarity_doc2vec_cluster(doc2vec_model, corpus_slice, ids_arxiv, i)

##################################################################################################

def read_corpus(texts, tokens_only=False):
    for i, line in enumerate(texts):
        try:
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags which is the document index
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
        except:
            pass


def doc2vec_train(train_corpus):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, workers=cores, dm=1, alpha=0.025, min_alpha=0.025)
    model.build_vocab(train_corpus)
    print("This model has a vocabulary of {} words".format(len(model.wv.vocab)))
    #starts training and decreases learning rates
    for epoch in range(10):
        print(epoch)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    print('finished training')
    return model


def get_similarity_doc2vec_cluster(doc2vec_model, corpus_slice, ids_arxiv, cluster_id):
    for i in range(len(corpus_slice)):
        # calculate similarity ALL vs. ALL within each cluster
        print(i)
        similarity_doc2vec = pd.DataFrame()
        v1 = corpus_slice[i].reshape(-1,1)
        #for j in np.arange(i+1, len(corpus)):
        for j in range(len(corpus_slice)):
            print(j)
            v2 = corpus_slice[i].reshape(-1,1)
            sim = distance.cosine(v1, v2)
            similarity_doc2vec = similarity_doc2vec.append(pd.Series([ids_arxiv.iloc[i], ids_arxiv.iloc[j], round(sim, 3)]), ignore_index=True)
        similarity_doc2vec.columns = ['ab1', 'ab2', 'sim_doc2vec']
        similarity_doc2vec.to_csv('similarity/similarity_doc2vec_cluster'+str(cluster_id)+'.csv', index=False)



distance.cosine(v1.reshape(-1,1), v1.reshape(-1,1))

def get_similarity_pairs_doc2vec(doc2vec_model):
    for i in range(len(doc2vec_model.docvecs)):
        print(i)
        inferred_vector = doc2vec_model.docvecs[i]
        sims = doc2vec_model.docvecs.most_similar([inferred_vector], topn=50)
        for id, sim in enumerate(sims, 1):
            line = "{'ab1':'" + str(clean_dataset.id[i]) + \
                    "','ab2':'" + str(clean_dataset.id[sim[0]]) + \
                    "','sim_doc2vec':" + str(round(sim[1], 3)) + "}\n"
            with open('data/similarity_doc2vec_1000.csv', 'a') as f:
                f.write(line)

def calculate_cluster_wm(doc2vec_model, corpus_slice, ids_arxiv, cluster_id):
    """ Word Mover’s Distance between two documents"""
    path2 = 'similarity/similarity_wm_topic_'+str(cluster_id)+'.csv'
    tot = int(len(corpus_slice) * len(corpus_slice) / 2 - len(corpus_slice))
    similarity_cluster_wm = pd.DataFrame({'cluster':np.zeros(tot), 'abs1':np.zeros(tot), 'abs2':np.zeros(tot), 'sim':np.zeros(tot)})
    current_line = 0
    for i in range(len(corpus_slice)):
        for j in np.arange(i + 1, len(corpus_slice)):
            similarity_cluster_wm.abs1[current_line] = ids_arxiv.iloc[i]
            similarity_cluster_wm.abs2[current_line] = ids_arxiv.iloc[j]
            current_line = current_line +1
    print('finished df')
    current_line = 0
    for i in range(len(corpus_slice)):
        print(i)
        for j in np.arange(i + 1, len(corpus_slice)):
            wm_dist = doc2vec_model.wv.wmdistance(corpus_slice[i], corpus_slice[j])
            similarity_cluster_wm.sim[current_line] = str(round(wm_dist, 3))
            current_line = current_line + 1
    similarity_cluster_wm.to_csv(path2)

def get_distance_wm(model):
    """ Word Mover’s Distance between two documents"""
    open(path2, "w").close()
    for i in np.arange(1, len(train_corpus)):
        print(i)
        abs1 = clean_dataset.id[i]
        print(datetime.datetime.now().time())
        for j in np.arange(i + 1, len(train_corpus)):
            abs2 = clean_dataset.id[j]
            wm_dist = model.wv.wmdistance(train_corpus[i].words, train_corpus[j].words)
            line = "{'ab1':'" + abs1 + \
                   "','ab2':'" + abs2 + \
                   "','wm_dist':" + str(round(wm_dist, 3)) + "}\n"
            with open(path2, 'a+') as f:
                f.write(line)


def assemble_dataset_from_files():
    dataset_doc2vec = pd.DataFrame()
    txt = open('data/similarity_doc2vec_10000.csv', 'r')
    for l in txt:
        l = l.replace("\'", "\"")
        print(l)
        line = json.loads(l)
        dataset_doc2vec = dataset_doc2vec.append(pd.Series([line['ab1'], line['ab2'], line['sim_doc2vec']]), ignore_index=True)
    dataset_doc2vec.columns = ['ab1', 'ab2', 'sim_doc2vec']
    dataset_doc2vec.to_csv("data/similarity_doc2vec_processed.csv", index=False)

def merge_similarities(file_sim1, file_sim2):
    pd_temp = pd.read_csv(file_sim1)
    print(pd_temp.head())
    similarities = pd.read_csv(file_sim2)
    similarities = similarities.merge(pd_temp, how="inner", on=['ab1', 'ab2'])
    similarities.to_csv('data/similarity_master.csv', index=False)



