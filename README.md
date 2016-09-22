# MarcBS/keras fork

This fork of Keras offers the following contributions:

- Caffe to Keras conversion module
- Layer-specific learning rates
- New layers for multimodal data

Contact email: marc.bolanos@ub.edu

GitHub page: https://github.com/MarcBS

MarcBS/keras is compatible with: __Python 2.7__ and __Theano__ only.

## Caffe to Keras conversion module

This module allows to convert Caffe models to Keras for their later training or test use.
See [this README](keras/caffe/README.md) for further information.

## Layer-specific learning rates

This functionality allows to add learning rates multipliers to each of the learnable layers in the networks. During training they will
be multiplied by the global learning rate for modifying the weight of the error on each layer independently. Here is a simple example of usage:

```
x = Dense(100, W_learning_rate_multiplier=10.0, b_learning_rate_multiplier=10.0)  (x)
```

## New layers for multimodal data

- [AttLSTM](https://github.com/MarcBS/keras/blob/ba642f5d345983c3ebeffede41c57e03a5c1f7ee/keras/layers/recurrent.py#L1261): LSTM with Attention mechanism.
- [LSTMCond](https://github.com/MarcBS/keras/blob/ba642f5d345983c3ebeffede41c57e03a5c1f7ee/keras/layers/recurrent.py#L940): LSTM with an additional input consisting in the previously generated word.

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
