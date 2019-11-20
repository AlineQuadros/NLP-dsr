# -*- coding: utf-8 -*-
import random
import numpy as np
import pickle
from keras import models
from preprocessing import encode_sequence


def decode_indices(indices, vocabulary):
    """ Decodes a sequence of indices and returns a string. """

    decoded_tokens = [vocabulary[index] for index in indices]
    return " ".join(decoded_tokens)
    

def generate_texts(text, preprocessed_corpus_path, model_path, log_path, generated_sequence_length, log=False):
    """ Generates a text from trained model """

    print("Generating texts...")

    # Getting all necessary data. That is the preprocessed corpus and the model.
    indices, vocabulary = pickle.load(open(preprocessed_corpus_path, "rb"))
    model = models.load_model(model_path)
    input_sequence = encode_sequence(text, vocabulary)
    # Generate a couple of texts.
    # Get a random temperature for prediction.
    temperature = random.uniform(0.0, 1.0)
    print("Temperature:", temperature)

    # Get a random sample as seed sequence.
    #random_index = random.randint(0, len(indices) - (generated_sequence_length))
    #input_sequence = encode_sequence(text, vocabulary)

    # Generate the sequence by repeatedly predicting.
    generated_sequence = []
    while len(generated_sequence) < generated_sequence_length:
        prediction = model.predict(np.expand_dims(input_sequence, axis=0))
        predicted_index = get_index_from_prediction(prediction[0], temperature)
        generated_sequence.append(predicted_index)
        input_sequence = input_sequence[1:]
        input_sequence.append(predicted_index)

    # Convert the generated sequence to a string.
    text = decode_indices(generated_sequence, vocabulary)
    if log:
        log_sample(log_path, text, model.get_config(), len(vocabulary))
    
    return text

        
def get_index_from_prediction(prediction, temperature=0.0):
    """ Gets an index from a prediction. """

    # Zero temperature - use the argmax.
    if temperature == 0.0:
        return np.argmax(prediction)

    # Non-zero temperature - do some random magic.
    else:
        prediction = np.asarray(prediction).astype('float64')
        prediction = np.log(prediction) / temperature
        exp_prediction= np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)
        probabilities = np.random.multinomial(1, prediction, 1)
        return np.argmax(probabilities)
    
def log_sample(log_path, text, model_config, vocabulary_len):
    """ adds to a log file that is keeping track of a sample input, 
    the generated text, and the model params used to """
    with open(log_path,'a') as log:
        line = str(model_config) + "," + str(vocabulary_len) + "," + str(text)
        log.write(line)