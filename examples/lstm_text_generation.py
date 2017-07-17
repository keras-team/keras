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
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
# convert to bytes
text = text.encode()
print('corpus length:', len(text))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen), dtype=np.uint8)
y = np.zeros((len(sentences)), dtype=np.uint8)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char
    y[i] = next_chars[i]


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Embedding(input_dim=256, output_dim=64))
model.add(LSTM(128))
model.add(Dense(256))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
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

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = b''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence.decode() + '"')
        sys.stdout.write(generated.decode())

        for i in range(400):
            x = np.zeros((1, maxlen))
            for t, char in enumerate(sentence):
                x[0, t] = char

            preds = model.predict(x, verbose=0)[0]
            next_byte = bytes([sample(preds, diversity)])

            generated += next_byte
            sentence = sentence[1:] + next_byte

            sys.stdout.write(next_byte.decode(errors='ignore'))
            sys.stdout.flush()
        print()
