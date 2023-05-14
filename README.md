# Keras Core: a new multi-backend Keras

Keras Core is a new multi-backend implementation of the Keras API, with support for TensorFlow, JAX, and PyTorch.

## Backwards compatibility

Keras Core is intend to work as a drop-in replacement for `tf.keras` (when using the TensorFlow backend).

## Why use Keras Core?

- Write custom components (e.g. layers, models, metrics) that you can move across framework boundaries.
- Make your code future-proof by avoiding framework lock-in.
- As a JAX user: get access to a fully-featured modeling and training library.
- As a PyTorch user: get access to the real Keras, at last!

## Credits

TODO
