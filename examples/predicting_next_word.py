'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
This script tries to predict next character. 
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical
import numpy as np
import random
import sys

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 1
step = 1
sentences = []
next_chars = []


for i in range(0, len(text)-1):
    sentences.append(char_indices[text[i:1+i]])
    next_chars.append(char_indices[text[i+1:i+2]])
    
print('nb sequences:', len(sentences))

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(Dense(128,input_dim=len(chars)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
p=0
seq_length = 25

for iteration in range(1, 600):
    print()
    print('-' * 50)
    print('Iteration', iteration)
       
    xs=to_categorical(sentences,len(chars))
    
    model.fit(xs, next_chars, nb_epoch=3,batch_size=seq_length)

    start_index = random.randint(0, len(text) - seq_length - 1)

    for diversity in [1.0]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + seq_length]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = []
            for j in range(len(sentence)):
                x +=[char_indices[ch] for ch in sentence[j]]
            x=to_categorical(x,len(chars))    
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()