# MarcBS/keras fork

![Build status](https://api.travis-ci.org/MarcBS/keras.svg)

This fork of Keras offers the following contributions:

- Caffe to Keras conversion module
- Layer-specific learning rates

Contact email: marc.bolanos@ub.edu

GitHub page: https://github.com/MarcBS

MarcBS/keras is compatible with: __Python 2.7__.

## Caffe to Keras conversion module

This module allows to convert Caffe models to Keras for their later training or test use.
See keras/caffe/README.md for further information.

## Layer-specific learning rates

This functionality allows to add learning rates multipliers to each of the learnable layers in the networks. During training they will
be multiplied by the global learning rate for modifying the weight of the error on each layer independently. Here is a simple example of usage:

```
model.add_node(Dense(100, W_learning_rate_multiplier=10.0, b_learning_rate_multiplier=10.0))
```

## Installation

In order to install the library you just have to follow these steps:

1) Clone this repository:
```
git clone https://github.com/MarcBS/keras.git
```
2) Include the repository path into your PYTHONPATH:
```
export PYTHONPATH=$PYTHONPATH:/path/to/keras
```


## Keras

For additional information on the Deep Learning library, visit the official web page www.keras.io or the GitHub repository https://github.com/fchollet/keras.
