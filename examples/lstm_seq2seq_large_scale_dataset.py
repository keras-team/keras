from __future__ import print_function

#import psutil
import os,datetime,time
import json
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import pickle
import traceback  
import gc
import time
import tensorflow as tf  
from keras import backend as K  
from keras.callbacks import TensorBoard


np.set_printoptions(threshold=np.inf)

batch_size = 128  # Batch size for training.

validate_batch_size = 1024

epochs = 1000  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
# Path to the data txt file on disk.
data_path = 'train_data.txt'
test_data_path = 'test_data.txt'
input_characters = set()
target_characters = set()
max_encoder_seq_length = 0
max_decoder_seq_length = 0

real_sample_size = 0

with open(data_path, 'r') as f:
    for line in f:
        if len(line.split('\t')) != 2:
            continue 
        input_text, target_text = line.split('\t')
        target_text = '\t' + target_text + '\n'
        if max_encoder_seq_length < len(input_text):
            max_encoder_seq_length = len(input_text)
        if max_decoder_seq_length < len(target_text):
            max_decoder_seq_length = len(target_text)
        for char in input_text:
            if char not in input_characters:
                   input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
        real_sample_size+=1


input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
print('Number of samples:', real_sample_size)

# input dict
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])

input_token_index_pkl_file = open('input_token_index.pkl','wb')
pickle.dump(input_token_index,input_token_index_pkl_file)
input_token_index_pkl_file.close()


# target dict
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

target_token_index_pkl_file = open('target_token_index.pkl','wb')
pickle.dump(target_token_index,target_token_index_pkl_file)
target_token_index_pkl_file.close()

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
           stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def generate_validation_data_from_file(path, batch_size):

    print(' initialize data frame')
    valid_encoder_input_data = np.zeros((validate_batch_size, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    valid_decoder_input_data = np.zeros((validate_batch_size, max_decoder_seq_length, num_decoder_tokens),dtype='float32')
    valid_decoder_target_data = np.zeros((validate_batch_size, max_decoder_seq_length, num_decoder_tokens),dtype='float32') 
    j = 0
    totalGenCnt = 0
    validdation_data_idx = 0
    while True:
        print(' enter generate_validation_data_from_file loop')
        with open(test_data_path) as f:
            for line in f:
                try:
                    if len(line.split('\t')) != 2:
                        continue
                    input_text, target_text = line.split('\t')
                    target_text = '\t' + target_text + '\n'
                    for t, char in enumerate(input_text):
                        if char == '\n':
                            continue
                        valid_encoder_input_data[j, t, input_token_index[char]] = 1.
                    for t, char in enumerate(target_text):
                        if char == '\n':
                            continue
                        # decoder_target_data is ahead of decoder_input_data by one timestep
                        valid_decoder_input_data[j, t, target_token_index[char]] = 1.
                        if t > 0:
                            # decoder_target_data will be ahead by one timestep
                            # and will not include the start character.
                            valid_decoder_target_data[j, t - 1, target_token_index[char]] = 1.
                    j+=1
                    if j == validate_batch_size:
                        j = 0
                        totalGenCnt = totalGenCnt + 1
                        print('\n'+str(totalGenCnt)+' generator validate input data size='+str(validate_batch_size)+' range from ('+str(validdation_data_idx)+','+str(validdation_data_idx+validate_batch_size)+')')
                        validdation_data_idx = validdation_data_idx + validate_batch_size
                        if validdation_data_idx > 10000 :
                        	validdation_data_idx = validdation_data_idx-10000
                        yield ([valid_encoder_input_data, valid_decoder_input_data],valid_decoder_target_data)
                        valid_encoder_input_data[:]=0
                        valid_decoder_input_data[:]=0
                        valid_decoder_target_data[:]=0
                except:
                    print("generate_validation_data_from_file exception "+str(j)+" input_text="+input_text)


def generate_train_data_from_file(path, batch_size):

    print(' initialize data frame')
    encoder_input_data = np.zeros((batch_size, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    decoder_input_data = np.zeros((batch_size, max_decoder_seq_length, num_decoder_tokens),dtype='float32')
    decoder_target_data = np.zeros((batch_size, max_decoder_seq_length, num_decoder_tokens),dtype='float32')    
    print(' enter generate_arrays_from_file')
    i = 0
    totalCallCnt = 0
    train_data_idx = 0
    while True:
        print(' enter generate_train_data_from_file loop')
        with open(data_path) as f:
            for line in f:
                try:
                    if len(line.split('\t')) != 2:
                        continue
                    input_text, target_text = line.split('\t')
                    for t, char in enumerate(input_text):
                        if char == '\n':
                            continue
                        encoder_input_data[i, t, input_token_index[char]] = 1.
                    for t, char in enumerate(target_text):
                        if char == '\n':
                            continue
                        # decoder_target_data is ahead of decoder_input_data by one timestep
                        decoder_input_data[i, t, target_token_index[char]] = 1.
                        if t > 0:
                            # decoder_target_data will be ahead by one timestep
                            # and will not include the start character.
                            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
                    i+=1
                    if i == batch_size:

                        i = 0
                        totalCallCnt = totalCallCnt + 1
                        train_data_idx = train_data_idx + batch_size
                        if train_data_idx > real_sample_size :
                        	train_data_idx = train_data_idx - real_sample_size
                        yield ([encoder_input_data, decoder_input_data],decoder_target_data)
                        encoder_input_data[:]=0
                        decoder_input_data[:]=0
                        decoder_target_data[:]=0
                except:
                    print("generate_train_data_from_file exception "+str(i)+" input_text="+input_text)

'''
epochs:
    The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached
max_queue_size:
    The default value for max_queue_size is 10, which is twice the number of your steps_per_epoch.
    The generator is called 10 times before the training begins to fill up the queue.
steps_per_epoch:
    It should typically be equal to the number of samples of your dataset divided by the batch size
validation_data:
    to evaluate the loss and any model metrics at the end of each epoch
validation_steps:
    It should typically be equal to the number of samples of your validation dataset divided by the batch size
'''
model.fit_generator(generate_train_data_from_file(data_path, batch_size),
          epochs=epochs,
          max_queue_size=10,
          steps_per_epoch=300,
          verbose=1,
          validation_data=generate_validation_data_from_file(test_data_path, batch_size),
          validation_steps=10)
print("model fit_generator")
# Save final model
model.save('s2s.h5')
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
