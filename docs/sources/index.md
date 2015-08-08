# Keras: Theano-based Deep Learning library

## Overview

Keras is a minimalist, highly modular neural network library in the spirit of Torch, written in Python, that uses [Theano](http://deeplearning.net/software/theano/) under the hood for optimized tensor manipulation on GPU and CPU. It was developed with a focus on enabling fast experimentation. 

Use Keras if you need a deep learning library that:

- allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
- supports both convolutional networks and recurrent networks, as well as combinations of the two.
- supports arbitrary connectivity schemes (including multi-input and multi-output training).
- runs seamlessly on CPU and GPU.

## Guiding principles

- __Modularity.__ A model is understood as a sequence or a graph of standalone, fully-configurable modules that can be plugged together with as little restrictions as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions, regularization schemes are all standalone modules that you can combine to create new models.

- __Minimalism.__ Each module should be kept short and simple (<100 lines of code). Every piece of code should be transparent upon first reading. No black magic: it hurts iteration speed and ability to innovate.

- __Easy extensibility.__ New modules are dead simple to add (as new classes/functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making Keras suitable for advanced research.

- __Work with Python__. No separate models configuration files in a declarative format (like in Caffe or PyLearn2). Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.

## Code

Find the code on Github: [fchollet/keras](https://github.com/fchollet/keras).

## License

Keras is licensed under the [MIT license](http://opensource.org/licenses/MIT). 

## Getting started: 30 seconds to Keras

The core datastructure of Keras is a __model__, a way to organize layers. There are two types of models: [`Sequential`](/models/#sequential) and [`Graph`](/models/#graph).

Here's the `Sequential` model (a linear pile of layers):

```python
from keras.models import Sequential

model = Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from keras.layers.core import Dense, Activation

model.add(Dense(input_dim=100, output_dim=64, init="glorot_uniform"))
model.add(Activation("relu"))
model.add(Dense(input_dim=64, output_dim=10, init="glorot_uniform"))
model.add(Activation("softmax"))
```

Once your model looks good, configure its learning process with `.compile()`:
```python
model.compile(loss='categorical_crossentropy', optimizer='sgd')
```

If you need to, you can further configure your optimizer. A core principle of Keras is make things things reasonably simple, while allowing the user to be fully in control when they need to (the ultimate control being the easy extensibility of the source code).
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
objective_score = model.evaluate(X_test, Y_test, batch_size=32)
```

Or generate predictions on new data:
```python
classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)
```

Building a network of LSTMs, a deep CNN, a Neural Turing Machine, a word2vec embedder or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be painful?

Have a look at the [examples](examples.md).

## Installation

Keras uses the following dependencies:

- __numpy__, __scipy__
- __pyyaml__
- __Theano__
    - See [installation instructions](http://deeplearning.net/software/theano/install.html#install).
- __HDF5__ and __h5py__ (optional, required if you use model saving/loading functions)
- Optional but recommended if you use CNNs: __cuDNN__.

Once you have the dependencies installed, clone the repo:
```bash
git clone https://github.com/fchollet/keras.git
```
Go to the Keras folder and run the install command:
```bash
cd keras
sudo python setup.py install
```
You can also install Keras from PyPI:
```
sudo pip install keras
```

## Support

You can ask questions and join the development discussion on the [Keras Google group](https://groups.google.com/forum/#!forum/keras-users).

## Contribution Guidelines

Keras welcomes all contributions from the community. 

- Keep a pragmatic mindset and avoid bloat. Only add to the source if that is the only path forward.
- New features should be documented. Make sure you update the documentation along with your Pull Request.
- The documentation for every new feature should include a usage example in the form of a code snippet. 
- All changes should be tested. Make sure any new feature you add has a corresponding unit test.
- Please no Pull Requests about coding style.
- Even if you don't contribute to the Keras source code, if you have an application of Keras that is concise and powerful, please consider adding it to our collection of [examples](https://github.com/fchollet/keras/tree/master/examples).


## Why this name, Keras?

Keras (κέρας) means _horn_ in Greek. It is a reference to a literary image from ancient Greek and Latin literature, first found in the _Odyssey_, where dream spirits (_Oneiroi_, singular _Oneiros_) are divided between those who deceive men with false visions, who arrive to Earth through a gate of ivory, and those who announce a future that will come to pass, who arrive through a gate of horn. It's a play on the words κέρας (horn) / κραίνω (fulfill), and ἐλέφας (ivory) / ἐλεφαίρομαι (deceive).

Keras was developed as part of the research effort of project __ONEIROS__ (*Open-ended Neuro-Electronic Intelligent Robot Operating System*).

> _"Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ 

> -- Homer, Odyssey 19. 562 ff (Shewring translation).
