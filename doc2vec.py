# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras import models
from keras import layers
from keras.utils import to_categorical
import os
from collections import Counter
from nltk import word_tokenize

from keras import utils

from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

import os
import gensim

# code from: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py

def read_corpus(texts, tokens_only=False):
    for i, line in enumerate(texts):
        try:
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
        except:
            pass
        


model = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=2, epochs=40)

model.build_vocab(train_corpus)
print(len(model.wv.vocab))
print(model.wv.vocab['crabs'].count)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

vector = model.infer_vector(['mangrove', 'plants', 'are', 'common', 'in', 'coasts'])
print(vector)

ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    # docvecs.most_similar calculates cosine similarity
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    #sims = model.docvecs.most_similar([inferred_vector])
    print("simil", sims)
    rank = [docid for docid, sim in sims].index(doc_id)
    print('ranks:', ranks)
    ranks.append(rank)

    second_ranks.append(sims[1])
    
import collections

counter = collections.Counter(ranks)
print(counter)

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
    
    
import random
doc_id = random.randint(0, len(train_corpus) - 1)

# Compare and print the second-most-similar document
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))