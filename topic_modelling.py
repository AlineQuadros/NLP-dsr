# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:46:19 2019
@author: Aline
Topic modelling
code adapted from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
"""
##############################################################################
import pandas as pd
import numpy as np
from nltk import tokenize
import glob
import json
import csv
import config

import re
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from  gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# tokenizer
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(clean_dataset.abstract))

print(data_words[:1])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words_nostops, min_count=5, threshold=5) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words_nostops], threshold=5)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words_nostops[651]]])
print(bigram_mod[data_words_nostops[454]])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatized[:2])
# Create Dictionary
# id2word = corpora.Dictionary(data_lemmatized)
id2word = corpora.Dictionary(data_words_bigrams)
# Create Corpus
texts = data_words_bigrams
# texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[345])

# Build LDA model

"""
class gensim.models.ldamodel.LdaModel(corpus=None, num_topics=100, id2word=None, distributed=False, chunksize=2000,
 passes=1, update_every=1, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, 
 gamma_threshold=0.001, minimum_probability=0.01, random_state=None, ns_conf=None, minimum_phi_value=0.01, 
 per_word_topics=False, callbacks=None, dtype=<class 'numpy.float32'>
"""
def get_LDA_topics_from_corpus(corpus, id2word, num_topics):
    """ extract n_topics from corpus using LDA """
    lda_model = LdaModel(corpus=corpus, id2word=id2word,num_topics=num_topics, random_state=100, update_every=5,
                         chunksize=100,passes=10, alpha='auto', per_word_topics=True)
    pprint(lda_model.print_topics())
    return lda_model


doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

def find_optimal_number_topics(corpus, id2word, min, max, step):
    range_coherence = {}
    range_perplexity = {}
    for n in np.arange(min, max, step):
        lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=20, random_state=100,
                             update_every=5, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        # Compute Perplexity - a measure of how good the model is. the lower the better
        perplexity_lda = lda_model.log_perplexity(corpus)
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        range_coherence.update({n: coherence_lda})
        range_perplexity.update({n: perplexity_lda})

    plt.plot(list(range_coherence.keys()), list(range_coherence.values()))
    plt.plot(list(range_perplexity.keys()), list(range_perplexity.values()))
    return range_coherence, range_perplexity