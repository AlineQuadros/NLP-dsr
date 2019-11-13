# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:26:36 2019

@author: MyLAP
"""

import preprocessing
import train_textgen


# define text source

# calculate stats

# set general parameters from stats
# Corpus parameters.
corpus_path = "abstracts.csv"

# Preprocessing parameters.
preprocessed_corpus_path = "abstracts_preprocessed.p"
most_common_words_number = 10000

# Training parameters.
model_path = "models/mangroves_model.h1"
dataset_size = 5000
sequence_length = 30
epochs = 1
batch_size = 128
hidden_size = 1000

# Generation parameters.
generated_sequence_length = 50


# prepare fro training
preprocessing.preprocess_corpus()

# train
train_textgen.train_neural_network()

# generate

generate_texts()