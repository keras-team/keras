
Here are a few examples to get you started!

### Multilayer Perceptron (MLP)

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
model.add(Dense(64, 2, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
score = model.evaluate(X_test, y_test, batch_size=16)
```

---

### Alternative implementation of MLP

```python
model = Sequential()
model.add(Dense(20, 64, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, 64, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, 2, init='uniform', activation='softmax')

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

---

### VGG-like convnet

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

---

### Sequence classification with LSTM

```python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
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

---

### Image captioning

Architecture for learning image captions with a convnet and a Gated Recurrent Unit (word-level embedding, caption of maximum length 16 words).

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

---

In the [examples folder](https://github.com/fchollet/keras/tree/master/examples), you will find example models for real datasets:

- CIFAR10 small images classification: Convnet with realtime data augmentation
- IMDB movie review sentiment classification: LSTM over sequences of words
- Reuters newswires topic classification: Multilayer Perceptron
