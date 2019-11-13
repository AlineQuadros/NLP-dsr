import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras import models
from keras import layers
from keras.utils import to_categorical
import os
from collections import Counter
import nltk
from nltk import word_tokenize
import pickle
import random
import keras
from keras import utils



def get_dataset(indices):

    """ Gets a full dataset of a defined size from the corpus. """

    print("Generating data set...")
    data_input = []
    data_output = []
    current_size = 0
    while current_size < dataset_size:

        # Randomly retrieve a sequence of tokens and the token right after it.
        random_index = random.randint(0, len(indices) - (sequence_length + 1))
        input_sequence = indices[random_index:random_index + sequence_length]
        output_sequence = indices[random_index + sequence_length]

        # Update arrays.
        data_input.append(input_sequence)
        data_output.append(output_sequence)

        # Next step.
        current_size += 1

    # Done. Return NumPy-arrays.
    data_input = np.array(data_input)
    data_output = np.array(data_output)
    return (data_input, data_output)

    

def train_neural_network():
    """ Trains the corpus """
    # Loading index-encoded corpus and vocabulary.
    indices, vocabulary = pickle.load(open(preprocessed_corpus_path, "rb"))

    # Get the dataset.
    print("Getting the dataset...")
    data_input, data_output = get_dataset(indices)
    data_output = utils.to_categorical(data_output, num_classes=len(vocabulary))

    # Creating the model.
    print("Creating model...")
    model = models.Sequential()
    model.add(layers.Embedding(len(vocabulary), hidden_size, input_length=sequence_length))
    model.add(layers.LSTM(hidden_size))
    model.add(layers.Dense(len(vocabulary)))
    model.add(layers.Activation('softmax'))
    model.summary()

    # Compining the model.
    print("Compiling model...")
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy']
    )

    # Training the model.
    print("Training model...")
    history = model.fit(data_input, data_output, epochs=epochs, batch_size=batch_size)
    model.save(model_path)
    #plot_history(history)
        

def plot_history(history):
    """ Plots the history of a training. """

    print(history.history.keys())

    # Render the loss.
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("history_loss.png")
    plt.clf()

    # Render the accuracy.
    plt.plot(history.history['categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("history_accuracy.png")
    plt.clf()

    plt.show()
    plt.close()
    