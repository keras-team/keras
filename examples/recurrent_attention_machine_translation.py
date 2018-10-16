'''Machine Translation (word-level) with recurrent attention in Keras.

This script demonstrates how to implement a basic word-level machine
translation using the recurrent attention mechanism described in [1].

# Summary of the algorithm
- ...
- ...
- In inference mode, TODO when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.


# Data
We use the machine translation dataset described in [2]. Note that this
is newer dataset that than what was used in [1].

To download the data run:
    mkdir -p data/wmt16_mmt
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/wmt16_mmt && rm training.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/wmt16_mmt && rm validation.tar.gz
    wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz && tar -xf mmt16_task1_test.tar.gz -C data/wmt16_mmt && rm mmt16_task1_test.tar.gz

# References
[1] Neural Machine Translation by Jointly Learning to Align and Translate
    https://arxiv.org/abs/1409.0473
[2] Multi30K: Multilingual English-German Image Descriptions
    https://arxiv.org/abs/1605.00459 (http://www.statmt.org/wmt16/multimodal-task.html)
'''

from __future__ import print_function

import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    RNN,
    GRUCell,
    TimeDistributed,
    Dense,
    concatenate)
from keras.models import Model
from keras.optimizers import Adadelta

# DATA_DIR = 'data/wmt16_mmt'
DATA_DIR = '/Users/andershuss/Datasets/WMT16/mmt'
FROM_LANGUAGE = 'en'
TO_LANGUAGE = 'de'

# Meta parameters
MAX_UNIQUE_WORDS = 30000
MAX_WORDS_PER_SENTENCE = 40  # inf in [1]
EMBEDDING_SIZE = 620  # `m` in [1]
RECURRENT_UNITS = 1000  # `n` in [1]
DENSE_ATTENTION_UNITS = 1000  # fixed equal to `n` in [1]
READOUT_HIDDEN_UNITS = 500  # `l` in [1]
OPTIMIZER = Adadelta(rho=0.95, epsilon=1e-6)

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.


# Load and tokenize the data.
def get_sentences(partion, language):
    fpath = os.path.join(DATA_DIR, partion + '.' + language)
    with open(fpath, 'r') as f:
        sentences = f.readlines()
    return ["<start> {} <end>".format(sentence) for sentence in sentences]


input_texts_train = get_sentences("train", FROM_LANGUAGE)
input_texts_val = get_sentences("val", FROM_LANGUAGE)
target_texts_train = get_sentences("train", TO_LANGUAGE)
target_texts_val = get_sentences("val", TO_LANGUAGE)

input_tokenizer = Tokenizer(num_words=MAX_UNIQUE_WORDS)
target_tokenizer = Tokenizer(num_words=MAX_UNIQUE_WORDS)
input_tokenizer.fit_on_texts(input_texts_train + input_texts_val)
target_tokenizer.fit_on_texts(target_texts_train + target_texts_val)

input_seqs_train = input_tokenizer.texts_to_sequences(input_texts_train)
input_seqs_val = input_tokenizer.texts_to_sequences(input_texts_val)
target_seqs_train = target_tokenizer.texts_to_sequences(target_texts_train)
target_seqs_val = target_tokenizer.texts_to_sequences(target_texts_val)

input_seqs_train, input_seqs_val, target_seqs_train, target_seqs_val = (
    pad_sequences(seq, maxlen=MAX_WORDS_PER_SENTENCE, padding='post')
    for seq in [input_seqs_train, input_seqs_val, target_seqs_train, target_seqs_val]
)

# build the model
x = Input((None,), name="input_sequences")
y = Input((None,), name="target_sequences")

x_emb = Embedding(input_tokenizer.num_words, EMBEDDING_SIZE)
y_emb = Embedding(target_tokenizer.num_words, EMBEDDING_SIZE)


class DenseAnnotationAttention():
    pass


encoder = Bidirectional(GRUCell(RECURRENT_UNITS))
x_enc = encoder(x_emb)

cell = DenseAnnotationAttention(cell=GRUCell(RECURRENT_UNITS),
                                units=DENSE_ATTENTION_UNITS,
                                output_attention_encoding=True)  # concatenates attention_h to cell output
decoder = RNN(cell, return_sequences=True)

h1 = decoder(y_emb, constants=x_enc)
h2 = TimeDistributed(Dense(READOUT_HIDDEN_UNITS))(concatenate([h1, y_emb]))  # TODO maxout
y_pred = TimeDistributed(Dense(target_tokenizer.num_words))


model = Model([y, x], y_pred)
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['acc'])

model.fit([target_seqs_train[:, :-1], input_seqs_train],
          target_seqs_train[:, 1:],
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(
              [target_seqs_train[:, :-1], input_seqs_train],
              target_seqs_train[:, 1:]))
