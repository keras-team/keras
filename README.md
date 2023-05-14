# Keras Core: a new multi-backend Keras

Keras Core is a new multi-backend implementation of the Keras API, with support for TensorFlow, JAX, and PyTorch.

## Backwards compatibility

Keras Core is intend to work as a drop-in replacement for `tf.keras` (when using the TensorFlow backend).

In addition, Keras models can consume datasets in any format, regardless of the backend you're using:
you can train your models with your existing tf.data.Dataset pipelines or Torch DataLoaders.

## Why use Keras Core?

- Write custom components (e.g. layers, models, metrics) that you can move across framework boundaries.
- Make your code future-proof by avoiding framework lock-in.
- As a PyTorch user: get access to power of Keras, at last!
- As a JAX user: get access to a fully-featured, battle-tested modeling and training library.

## Credits

TODO
