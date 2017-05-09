'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('corpus length:', len(text))
# chars is all char dict
chars = sorted(list(set(text)))
print('total chars:', len(chars))
# dict for char to index
char_indices = dict((c, i) for i, c in enumerate(chars))
# dict for index to char
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
# each training data length is 40
maxlen = 40
# window size for sliding
step = 3
# input for train data
sentences = []
# output for train data
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
# X: 3D matrix 1: train data size 2:max sentence length 3:one-hot vector for each char in sentence
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
# y: 2D matrix 1: train data size 2:one-hot vector for each char to predict(output)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
# fullfill X & y
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
# hidden layer unit num=128
# LSTM(output_dim=hidden_neurons, input_dim=in_neurons, return_sequences=False)
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
# after the first layer, you don't need to specify the size of the input anymore
# because the input size must the output size of last layer
# len(chars) is the final output size which will give the prob for one-hot vector
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=1)

    start_index = random.randint(0, len(text) - maxlen - 1)
    
    # diversity: sample seed for select char from output vector
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
	# sentence as seed
        sentence = text[start_index: start_index + maxlen]
        
	generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
	    # sentence slide by one char each time for next predict loop
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
