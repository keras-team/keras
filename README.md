# Keras Core: a new multi-backend Keras

Keras Core is a new multi-backend implementation of the Keras API, with support for TensorFlow, JAX, and PyTorch.

## Local installation

Run `python3 pip_build.py --install` from the root directory.

## Backwards compatibility

Keras Core is intend to work as a drop-in replacement for `tf.keras` (when using the TensorFlow backend).

If your `tf.keras` model does not include custom compoments, you can start running it on top of JAX or PyTorch immediately.

If it does include custom components (e.g. custom layers or a custom `train_step()`), it is usually possible to convert it
to a backend-agnostic implementation in just a few minutes.

In addition, Keras models can consume datasets in any format, regardless of the backend you're using:
you can train your models with your existing tf.data.Dataset pipelines or Torch DataLoaders.

## Why use Keras Core?

- Run your high-level Keras workflows on top of any framework -- benefiting at will from the advantages of each framework,
e.g. the scalability and performance of JAX or the production ecosystem options of TensorFlow.
- Write custom components (e.g. layers, models, metrics) that you can use in low-level workflows in any framework.
    - You can take a Keras model and train it in a training loop written from scratch in native TF, JAX, or PyTorch.
    - You can take a Keras model and use it as part of a PyTorch-native `Module` or as part of a JAX-native model function.
- Make your ML code future-proof by avoiding framework lock-in.
- As a PyTorch user: get access to power and usability of Keras, at last!
- As a JAX user: get access to a fully-featured, battle-tested, well-documented modeling and training library.

## Credits

TODO
