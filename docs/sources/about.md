# Keras: Theano-based Deep Learning library

## Overview

Keras is a minimalist, highly modular neural network library in the spirit of Torch, written in Python/Theano. It was developed with a focus on enabling fast experimentation. 

## Guiding principles

- __Modularity.__ A model is understood as a sequence of standalone, fully-configurable modules that can be plugged together with as little restrictions as possible. In particular, neural layers, cost functions, optimizers, initialization schemes, activation functions and dropout are all standalone modules that you can combine to create new models. 

- __Minimalism.__ Each module should be kept short and simple (<100 lines of code). Every piece of code should be transparent upon first reading. No black magic: it hurts iteration speed and ability to innovate. 

- __Easy extensibility.__ A new feature (a new module, per the above definition, or a new way to combine modules together) are dead simple to add (as new classes/functions), and existing modules provide ample examples.

- __Work with Python__. No separate models configuration files in a declarative format (like in Caffe or PyLearn2). Models are described in Python code, which is compact, easier to debug, benefits from syntax highlighting, and most of all, allows for ease of extensibility.


## Contribution Guidelines

Keras welcomes all contributions from the community. 

- Keep a pragmatic mindset and avoid bloat. Only add to the source if that is the only path forward. Every additional line of code is a liability.
- New features should be documented. Make sure you update the documentation along with your Pull Request.
- The documentation for every new feature should include a usage example in the form of a code snippet. 
- All changes should be tested. A formal test process will be introduced very soon.
- Even if you don't contribute to the Keras source code, if you have an application of Keras that is concise and powerful, please consider adding to our collection of [examples](https://github.com/fchollet/keras/tree/master/examples).