# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:35:56 2019

@author: MyLAP
"""

def decode_indices(indices, vocabulary):
    """ Decodes a sequence of indices and returns a string. """

    decoded_tokens = [vocabulary[index] for index in indices]
    if use_moses_detokenizer  == True:
        return detokenizer.detokenize(decoded_tokens, return_str=True)
    else:
        return " ".join(decoded_tokens)
    

def generate_texts(text):
    """ Generates a couple of random texts. """

    print("Generating texts...")

    # Getting all necessary data. That is the preprocessed corpus and the model.
    indices, vocabulary = pickle.load(open(preprocessed_corpus_path, "rb"))
    model = models.load_model(model_path)

    # Generate a couple of texts.
    for _ in range(10):

        # Get a random temperature for prediction.
        temperature = random.uniform(0.0, 1.0)
        print("Temperature:", temperature)

        # Get a random sample as seed sequence.
        #random_index = random.randint(0, len(indices) - (generated_sequence_length))
        input_sequence = encode_sequence(text, vocabulary)

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
        print(text)
        print("")

        
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