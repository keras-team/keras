# Keras: Theano-based Deep Learning library

## You have just found Keras.

Keras is a minimalist, highly modular neural network library in the spirit of Torch, written in Python / Theano so as not to have to deal with the dearth of ecosystem in Lua. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:
- allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
- supports both convolutional networks (for vision) and recurrent networks (for sequence data). As well as combinations of the two. 
- runs seamlessly on the CPU and the GPU.

Read the documentation at [Keras.io](http://keras.io).

Keras is compatible with __Python 2.7-3.4__.

## Guiding principles

- __Modularity.__ A model is understood as a sequence of standalone, fully-configurable modules that can be plugged together with as little restrictions as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions and dropout are all standalone modules that you can combine to create new models. 

- __Minimalism.__ Each module should be kept short and simple (<100 lines of code). Every piece of code should be transparent upon first reading. No black magic: it hurts iteration speed and ability to innovate. 

- __Easy extensibility.__ New features (a new module, per the above definition, or a new way to combine modules together) are dead simple to add (as new classes/functions), and existing modules provide ample examples.

- __Work with Python__. No separate models configuration files in a declarative format (like in Caffe or PyLearn2). Models are described in Python code, which is compact, easier to debug, benefits from syntax highlighting, and most of all, allows for ease of extensibility. See for yourself with the examples below.

## Examples

### Multilayer Perceptron (MLP):

```python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(20, 64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, 64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, 1, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
score = model.evaluate(X_test, y_test, batch_size=16)
```

### Alternative implementation of MLP:

```python
model = Sequential()
model.add(Dense(20, 64, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, 64, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, 1, init='uniform', activation='softmax')

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

### VGG-like convnet:

```python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

model = Sequential()
model.add(Convolution2D(32, 3, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(32, 32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 32, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(64, 64, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64*8*8, 256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256, 10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)

```

### Sequence classification with LSTM:

```python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Embedding
from keras.layers.recurrent import LSTM

model = Sequential()
model.add(Embedding(max_features, 256))
model.add(LSTM(256, 128, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
score = model.evaluate(X_test, Y_test, batch_size=16)
```

### Architecture for learning image captions with a convnet and a Gated Recurrent Unit:
(word-level embedding, caption of maximum length 16 words).

Note that getting this to actually "work" will require using a bigger convnet, initialized with pre-trained weights.
Displaying readable results will also require an embedding decoder.

```python
max_caption_len = 16

model = Sequential()
model.add(Convolution2D(32, 3, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(32, 32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))

model.add(Convolution2D(64, 32, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(64, 64, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))

model.add(Convolution2D(128, 64, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(128, 128, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))

model.add(Flatten())
model.add(Dense(128*4*4, 256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Repeat(max_caption_len)) 
# the GRU below returns sequences of max_caption_len vectors of size 256 (our word embedding size)
model.add(GRU(256, 256, return_sequences=True))

model.compile(loss='mean_squared_error', optimizer='rmsprop')

# "images" is a numpy array of shape (nb_samples, nb_channels=3, width, height) 
# "captions" is a numpy array of shape (nb_samples, max_caption_len=16, embedding_dim=256)
# captions are supposed already embedded (dense vectors).
model.fit(images, captions, batch_size=16, nb_epoch=100)
    
```

In the examples folder, you will find example models for real datasets:
- CIFAR10 small images classification: Convnet with realtime data augmentation
- IMDB movie review sentiment classification: LSTM over sequences of words
- Reuters newswires topic classification: Multilayer Perceptron


## Current capabilities

For complete coverage of the API, check out [the Keras documentation](http://keras.io).

A few highlights: convnets, LSTM, GRU, word2vec-style embeddings, PReLU, batch normalization...

## Installation

Keras uses the following dependencies:

- numpy, scipy

- Theano
    - See installation instructions: http://deeplearning.net/software/theano/install.html#install

- h5py (optional, required if you use model saving/loading functions)

- Optional but recommended if you use CNNs: cuDNN.

Once you have the dependencies installed, cd to the Keras folder and run the install command:
```
sudo python setup.py install
```

## Why this name, Keras?

Keras (κέρας) means _horn_ in Greek. It is a reference to a literary image from ancient Greek and Latin literature, first found in the _Odyssey_, where dream spirits (_Oneiroi_, singular _Oneiros_) are divided between those who deceive men with false visions, who arrive to Earth through a gate of ivory, and those who announce a future that will come to pass, who arrive through a gate of horn. It's a play on the words κέρας (horn) / κραίνω (fulfill), and ἐλέφας (ivory) / ἐλεφαίρομαι (deceive).

Keras was developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System).

_"Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ Homer, Odyssey 19. 562 ff (Shewring translation).

