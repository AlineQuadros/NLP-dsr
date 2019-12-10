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
import joblib
from operator import itemgetter
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from  gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt

# spacy for lemmatization
import en_core_web_lg
nlp = en_core_web_lg.load()
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

############################################################################3

clean_dataset = pd.read_csv('app/data/clean_dataset.csv')

def LDA_preparation_pipeline(texts):
    # tokenizer (STEP 1)
    corpus_tokenized = list(sent_to_words(clean_dataset.abstract))
    joblib.dump(corpus_tokenized, 'corpus/all_abstracts_tokenized.cor')

    # Remove Stop Words (STEP 2)
    corpus_tokenized_ready = remove_stopwords(corpus_tokenized)
    joblib.dump(corpus_tokenized_ready, 'corpus/all_abstracts_stop_removed.cor')

    # lemmatization (STEP 3)
    corpus_tokenized_ready = apply_lemmatization(corpus_tokenized_ready)
    joblib.dump(corpus_tokenized_ready, 'corpus/all_abstracts_lemmatized.cor')

    # Build the bigram and trigram models (STEP 4)
    bigram = gensim.models.Phrases(corpus_tokenized_ready, min_count=5, threshold=4) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # print(bigram_mod[corpus_tokenized_ready[33]])
    corpus_tokenized_ready = make_bigrams(corpus_tokenized_ready)
    joblib.dump(corpus_tokenized_ready, 'corpus/all_abstracts_with_bigrams.cor')

    # Create Dictionary (STEP 5)
    id2word = corpora.Dictionary(corpus_tokenized_ready)
    joblib.dump(id2word, 'corpus/dictionary_for_lda.cor')

    # Create a corpus with a BOW (STEP 6)
    corpus = [id2word.doc2bow(text) for text in corpus_tokenized_ready]
    joblib.dump(corpus, 'corpus/corpus_bow_for_lda.cor')

    # View
    #print(corpus[0])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
# Define functions for stopwords, bigrams, trigrams and lemmatization

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def apply_lemmatization(texts):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc])
    return texts_out

# Build LDA model
"""
class gensim.models.ldamodel.LdaModel(corpus=None, num_topics=100, id2word=None, distributed=False, chunksize=2000,
 passes=1, update_every=1, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, 
 gamma_threshold=0.001, minimum_probability=0.01, random_state=None, ns_conf=None, minimum_phi_value=0.01, 
 per_word_topics=False, callbacks=None, dtype=<class 'numpy.float32'>
"""
def get_LDA_topics_from_corpus(corpus, id2word, num_topics):
    """ extract n_topics from corpus using LDA """
    lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100, update_every=5,
                         chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    pprint(lda_model.print_topics())
    return lda_model


doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

range_coherence = {}
range_perplexity = {}
def find_optimal_number_topics(corpus, id2word, min, max, step, texts):
    for n in np.arange(min, max, step):
        lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=n, random_state=100,
                             update_every=5, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        # Compute Perplexity - a measure of how good the model is. the lower the better
        perplexity_lda = lda_model.log_perplexity(corpus)
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        range_coherence.update({n: coherence_lda})
        range_perplexity.update({n: perplexity_lda})

    plt.plot(list(range_coherence.keys()), list(range_coherence.values()))
    plt.plot(list(range_perplexity.keys()), list(range_perplexity.values()))
    return range_coherence, range_perplexity

# Save model to disk
lda_model.save('models/LDA_10topics')

# Load a potentially pretrained model from disk.
lda = LdaModel.load('models/LDA_20topics')

def get_best_topic(corpus):
    topic_abstract_lda_10 = pd.DataFrame(columns=['id', 'topic'])
    for i in range(len(corpus)):
        topic_abstract_lda_10 = topic_abstract_lda_10.append({'id': clean_dataset.id[i], 'topic': max(lda_model.get_document_topics(corpus[i]), key=itemgetter(1))[0]}, ignore_index=True)
    topic_abstract_lda_10.to_csv('processed/topic_abstract_lda_10.csv', index=False)
    return topic_abstract_lda_10


