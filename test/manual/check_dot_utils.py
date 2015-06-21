from keras.utils.dot_utils import Grapher

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU

ent_lookup = Sequential()
ent_lookup.add(Embedding(10, 2))
ent_lookup.add(Flatten())

rel_lookup = Sequential()
rel_lookup.add(Embedding(20, 2))
rel_lookup.add(Flatten())

word_sequence = Sequential()
word_sequence.add(Embedding(10, 5))
word_sequence.add(GRU(5, 2))

model = Sequential()
model.add(Merge([word_sequence, ent_lookup, rel_lookup], mode='concat'))
model.add(Activation('relu'))
model.add(Dense(6, 2))
model.add(Activation('softmax'))

g = Grapher()
g.plot(model, 'mymodel.png')