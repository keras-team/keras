# Keras: Deep Learning library for Theano and TensorFlow

[![Build Status](https://travis-ci.org/fchollet/keras.svg?branch=master)](https://travis-ci.org/fchollet/keras)
[![PyPI version](https://badge.fury.io/py/keras.svg)](https://badge.fury.io/py/keras)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fchollet/keras/blob/master/LICENSE)

## You have just found Keras.

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either [TensorFlow](https://github.com/tensorflow/tensorflow) or [Theano](https://github.com/Theano/Theano). It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

- allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
- supports both convolutional networks and recurrent networks, as well as combinations of the two.
- supports arbitrary connectivity schemes (including multi-input and multi-output training).
- runs seamlessly on CPU and GPU.

Read the documentation at [Keras.io](http://keras.io).

Keras is compatible with: __Python 2.7-3.5__.


------------------


## Guiding principles

- __Modularity.__ A model is understood as a sequence or a graph of standalone, fully-configurable modules that can be plugged together with as little restrictions as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions, regularization schemes are all standalone modules that you can combine to create new models.

- __Minimalism.__ Each module should be kept short and simple. Every piece of code should be transparent upon first reading. No black magic: it hurts iteration speed and ability to innovate.

- __Easy extensibility.__ New modules are dead simple to add (as new classes and functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making Keras suitable for advanced research.

- __Work with Python__. No separate models configuration files in a declarative format. Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.


------------------


## Getting started: 30 seconds to Keras

The core data structure of Keras is a __model__, a way to organize layers. The main type of model is the [`Sequential`](http://keras.io/getting-started/sequential-model-guide) model, a linear stack of layers. For more complex architectures, you should use the [Keras functional API](http://keras.io/getting-started/functional-api-guide).

Here's the `Sequential` model:

```python
from keras.models import Sequential

model = Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from keras.layers import Dense, Activation

model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
```

Once your model looks good, configure its learning process with `.compile()`:
```python
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

If you need to, you can further configure your optimizer. A core principle of Keras is to make things reasonably simple, while allowing the user to be fully in control when they need to (the ultimate control being the easy extensibility of the source code).
```python
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```

You can now iterate on your training data in batches:
```python
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
```

Alternatively, you can feed batches to your model manually:
```python
model.train_on_batch(X_batch, Y_batch)
```

Evaluate your performance in one line:
```python
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
```

Or generate predictions on new data:
```python
classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)
```

Building a question answering system, an image classification model, a Neural Turing Machine, a word2vec embedder or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be painful?

For a more in-depth tutorial about Keras, you can check out:

- [Getting started with the Sequential model](http://keras.io/getting-started/sequential-model-guide)
- [Getting started with the functional API](http://keras.io/getting-started/functional-api-guide)

In the [examples folder](https://github.com/fchollet/keras/tree/master/examples) of the repository, you will find more advanced models: question-answering with memory networks, text generation with stacked LSTMs, etc.


------------------


## Installation

Keras uses the following dependencies:

- numpy, scipy
- pyyaml
- HDF5 and h5py (optional, required if you use model saving/loading functions)
- Optional but recommended if you use CNNs: cuDNN.

*When using the Theano backend:*

- Theano
    - [See installation instructions](http://deeplearning.net/software/theano/install.html#install).

*When using the TensorFlow backend:*

- TensorFlow
    - [See installation instructions](https://github.com/tensorflow/tensorflow#download-and-setup).

To install Keras, `cd` to the Keras folder and run the install command:
```
sudo python setup.py install
```

You can also install Keras from PyPI:
```
sudo pip install keras
```

------------------


## Switching from Theano to TensorFlow

By default, Keras will use Theano as its tensor manipulation library. [Follow these instructions](http://keras.io/backend/) to configure the Keras backend.

------------------


## Support

You can ask questions and join the development discussion on the [Keras Google group](https://groups.google.com/forum/#!forum/keras-users).

You can also post bug reports and feature requests in [Github issues](https://github.com/fchollet/keras/issues). Make sure to read [our guidelines](https://github.com/fchollet/keras/blob/master/CONTRIBUTING.md) first.


------------------


## Why this name, Keras?

Keras (κέρας) means _horn_ in Greek. It is a reference to a literary image from ancient Greek and Latin literature, first found in the _Odyssey_, where dream spirits (_Oneiroi_, singular _Oneiros_) are divided between those who deceive men with false visions, who arrive to Earth through a gate of ivory, and those who announce a future that will come to pass, who arrive through a gate of horn. It's a play on the words κέρας (horn) / κραίνω (fulfill), and ἐλέφας (ivory) / ἐλεφαίρομαι (deceive).

Keras was initially developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System).

>_"Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ Homer, Odyssey 19. 562 ff (Shewring translation).

------------------
