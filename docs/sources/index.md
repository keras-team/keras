# Keras: Python/Theano Deep Learning library

## Overview

Keras is a minimalist, highly modular neural network library in the spirit of Torch, written in Python, and using [Theano](http://deeplearning.net/software/theano/) for fast tensor manipulation on GPU and CPU. It was developed with a focus on enabling fast experimentation. 

Use Keras if you need a deep learning library that:

- allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
- supports both __convolutional networks__ and __recurrent networks__ (LSTM, GRU, etc). As well as combinations of the two. 
- runs seamlessly on the CPU and the GPU.

## Guiding principles

- __Modularity.__ A model is understood as a sequence of standalone, fully-configurable modules that can be plugged together with as little restrictions as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions and dropout are all standalone modules that you can combine to create new models. 

- __Minimalism.__ Each module should be kept short and simple (<100 lines of code). Every piece of code should be transparent upon first reading. No black magic: it hurts iteration speed and ability to innovate. 

- __Easy extensibility.__ A new feature (a new module, per the above definition, or a new way to combine modules together) are dead simple to add (as new classes/functions), and existing modules provide ample examples.

- __Work with Python__. No separate models configuration files in a declarative format (like in Caffe or PyLearn2). Models are described in Python code, which is compact, easier to debug, benefits from syntax highlighting, and most of all, allows for ease of extensibility.

## Code

Find the code on Github: [fchollet/keras](https://github.com/fchollet/keras).

## License

Keras is licensed under the [MIT license](http://opensource.org/licenses/MIT). 

## Getting started: 30 seconds to Deep Learning with Keras

The core datastructure of Keras is a __model__, a way to organize layers. Here's a sequential model (a linear pile of layers).

```python
from keras.models import Sequential

model = Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from keras.layers.core import Dense, Activation

model.add(Dense(input_dim=100, output_dim=64, init="uniform"))
model.add(Activation("relu"))
model.add(Dense(input_dim=64, output_dim=10, init="uniform"))
model.add(Activation("softmax"))
```

Once your model looks good, configure its learning process with `.compile()`:
```python
model.compile(loss='categorical_crossentropy', optimizer='sgd')
```

Alernatively, further configure your optimizer:
```python
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```

You can now iterate on your learning data in batches:
```python
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
```

Alternatively, you can feed batches to your model manually:
```python
model.fit(X_batch, Y_batch)
```

Evaluate your performance in one line:
```python
objective_score = model.evaluate(X_test, Y_test, batch_size=32)
```

Or just generate predictions on new data:
```python
classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)
```

Building a network of LSTMs, a deep CNN, a word2vec embedder or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be complicated?

Have a look at the [examples](examples.md).


## Contribution Guidelines

Keras welcomes all contributions from the community. 

- Keep a pragmatic mindset and avoid bloat. Only add to the source if that is the only path forward. Every additional line of code is a liability.
- New features should be documented. Make sure you update the documentation along with your Pull Request.
- The documentation for every new feature should include a usage example in the form of a code snippet. 
- All changes should be tested. A formal test process will be introduced very soon.
- Even if you don't contribute to the Keras source code, if you have an application of Keras that is concise and powerful, please consider adding it to our collection of [examples](https://github.com/fchollet/keras/tree/master/examples).

