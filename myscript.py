# -*- coding: utf-8 -*-
#from preprocessing import preprocess_corpus
#from train_textgen import train_neural_network
#from generate import generate_texts

import spacy
import pandas as pd
import config

# define text source
#python -m spacy download en_core_web_sm

#nlp = spacy.load("en_core_web_sm")
def printme():
        print('hello')

if __name__ == '__main__':
        printme()




# calculate stats
# --------------------- declare global vars ----------------------------------
# set general parameters from stats
# Corpus parameters



def text_generation():
        titles = pd.read_csv('data/titles.csv')
        abstracts = pd.read_csv('data/abstracts.csv')
        corpus_path = "data/all_text.txt"
        # TODO fix this funct
        # Preprocessing parameters
        preprocessed_corpus_path = "processed/abstracts_preprocessed.p"
        most_common_words_number = 10000

        # Training parameters
        model_path = "models/mangroves_model.h1"
        dataset_size = 5000
        sequence_length = 30
        epochs = 10
        batch_size = 128
        hidden_size = 1000
        # Generation parameters.
        generated_sequence_length = 30
        log_path = "logs/log_generated_texts.csv"
        # prepare for training
        preprocess_corpus()
        # train
        train_neural_network(preprocessed_corpus_path, model_path, hidden_size, sequence_length, epochs, batch_size)
        # generate
        text = ["mangrove", "crabs", "have", "an", "important", "role", "in", "the", "mangroves",
                "of", "the", "Pacific", "ocean", "near", "India", "where", "they", "live",
                "near", "rhizophora", "plants", "and", "feed", "from", "their", "leaves",
                "and", "also", "they", "also", "predators", "invetebrates"]
        generate_texts(text, preprocessed_corpus_path, model_path, log_path, generated_sequence_length, log=True)
