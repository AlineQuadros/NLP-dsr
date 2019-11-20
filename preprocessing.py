# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import spacy
import gensim
import pickle
from collections import Counter
import nltk
from nltk import word_tokenize

preprocess_anyway = True

# extract basic stats to understand the corpus
# mean sentence length
# mean number sentences per doc
# mean number words per doc

def add_stats(texts):
    """ receives a series of texts and returns dataframe with stats"""
    
    corpus_stats = pd.DataFrame(columns=["sentence_count", "word_count", "mean_word_count"])
    for a, i in texts.item():
        sentences = nltk.sent_tokenize(a)
        corpus_stats.loc[i, "sentence_count"] = len(sentences)
        corpus_stats.loc[i, "word_count"] = 33 #TODO
                
        temp = []
        for s in sentences:
            tokens = nltk.word_tokenize(s)
            temp = temp.append(len(tokens))
        corpus_stats.loc[i, "mean_word_count"] = np.mean(temp)
    return corpus_stats

def make_fulltext(texts, corpus_path):
    fulltext = open(corpus_path, "w")
    for t in texts:
        try:         
            fulltext.write(t)
        except:
            next
    fulltext.close()
    return fulltext
    

def encode_sequence(sequence, vocabulary):
    """ Encodes a sequence of tokens into a sequence of indices. """

    return [vocabulary.index(element) for element in sequence if element in vocabulary]


def preprocess_corpus(corpus_path, preprocessed_corpus_path, most_common_words_number):
    """ loads a text file and generate indices adn vocabulary from tokens"""

    if preprocess_anyway == True:
        print("Preprocessing corpus...")

        # Opening the file.
        corpus_file = open(corpus_path, "r")
        corpus_string = corpus_file.read()

        # Getting the vocabulary.
        print("Tokenizing...")
        corpus_tokens = word_tokenize(corpus_string)
        print("Number of tokens:", len(corpus_tokens))
        print("Building vocabulary...")
        word_counter = Counter()
        word_counter.update(corpus_tokens)
        print("Length of vocabulary before pruning:", len(word_counter))
        vocabulary = [key for key, value in word_counter.most_common(most_common_words_number)]
        print("Length of vocabulary after pruning:", len(vocabulary))

        # Converting to indices.
        print("Index-encoding...")
        indices = encode_sequence(corpus_tokens, vocabulary)
        print("Number of indices:", len(indices))

        # Saving.
        print("Saving file...")
        pickle.dump((indices, vocabulary), open(preprocessed_corpus_path, "wb"))
    else:
        print("Corpus already preprocessed.")
        

#doc = nlp(abstracts.abstract[i])
#for token in doc:
#    if token.pos_ == "PROPN":
#        print(token.text, token.pos_, token.dep_)

    # print(token.text, token.pos_, token.dep_)
