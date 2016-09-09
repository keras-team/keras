'''This example demonstrates the use of fasttext for text classification

Based on Joulin et al's paper:

Bags of Tricks for Efficient Text Classification
https://arxiv.org/abs/1607.01759

Results on IMDB datasets with uni/bi-gram embeddings:
    Uni-gram: 0.8813 test accuracy after 5 epochs. 15s/epoch on i7 cpu.
    Bi-gram : 0.9019 test accuracy after 5 epochs. 5s/epoch on GTX 1080 gpu.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Embedding
from keras.layers import AveragePooling1D
from keras.datasets import imdb


def create_ngram_set(input_list, ngram_range=2):
    """
    Extract n-grams from a list of integers

    >>> create_ngram_set([1, 4, 9, 4, 1], ngram_range=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    :param input_list:
    :param ngram_range: integer
    :return: set of tuples
    """
    return set(zip(*[input_list[i:] for i in range(ngram_range)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the list of sequences by appending bi-grams terms

    >>> print(add_ngram([[1, 3, 4, 5], [1, 3, 7, 9, 2]],
        token_indice={(1, 3): 1337, (9, 2): 42, (4, 5): 2017},
        ngram_range=2))
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    :param sequences:
    :param token_indice:
    :param ngram_range:
    :return:
    """
    augmented_sequences = []

    for sequence in sequences.copy():
        for i in range(len(sequence)-ngram_range+1):
            ngram = tuple(sequence[i:i+ngram_range])
            if ngram in token_indice:
                sequence.append(token_indice[ngram])
        augmented_sequences.append(sequence)

    return augmented_sequences


# set parameters:
UNIGRAM = True
BIGRAM = False
max_features = 20000
maxlen = 400
batch_size = 32
embedding_dims = 20
nb_epoch = 5

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

if BIGRAM:

    # Limit the maximum number of bi-gram to max_gram_index
    ngram_range = 2
    max_ngram_index = 1200000

    # Create n-gram_alphabet from the training set (X_train)
    ngram_set = set()
    for input_list in X_train:
        set_of_ngram = create_ngram_set(input_list)
        ngram_set.update(set_of_ngram)

    # Creation dictionary mapping a n-gram token to a integer
    # the integer value is greater that max_features:
    # the number of features when we extracted the data
    start_index = max_features + 1
    token_indice = {v: k+start_index for k, v in enumerate(ngram_set) if k+start_index < max_ngram_index}
    indice_token = {token_indice[k]: k for k in token_indice}

    # Augmenting X_train, X_test with n-grams features
    X_train = add_ngram(X_train, token_indice, ngram_range)
    X_test = add_ngram(X_test, token_indice, ngram_range)

    # Set the highest integer that could be found in the dataset
    max_features = max_ngram_index + 1


print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))

# we add a AveragePooling1D, which will average the embeddings
# of all words in the document
model.add(AveragePooling1D(pool_length=model.output_shape[1]))

# We flatten the output of the AveragePooling1D layer
model.add(Flatten())

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
