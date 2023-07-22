# Keras Core: A new multi-backend Keras

Keras Core is a new multi-backend implementation of the Keras API, with support for TensorFlow, JAX, and PyTorch.

**WARNING:** At this time, this package is experimental.
It has rough edges and not everything might work as expected.
We are currently hard at work improving it.

Once ready, this package will become Keras 3.0 and subsume `tf.keras`.

## Local installation

Keras Core is compatible with Linux and MacOS systems. To install a local development version:

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run installation command from the root directory.

```
python pip_build.py --install
```

You should also install your backend of choice: `tensorflow`, `jax`, or `torch`.
Note that `tensorflow` is required for using certain Keras Core features: certain preprocessing layers as
well as `tf.data` pipelines.

## Configuring your backend

You can export the environment variable `KERAS_BACKEND` or you can edit your local config file at `~/.keras/keras.json`
to configure your backend. Available backend options are: `"tensorflow"`, `"jax"`, `"torch"`. Example:

```
export KERAS_BACKEND="jax"
```

In Colab, you can do:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras_core as keras
```

## Backwards compatibility

Keras Core is intended to work as a drop-in replacement for `tf.keras` (when using the TensorFlow backend). Just take your
existing `tf.keras` code, change the `keras` imports to `keras_core`, make sure that your calls to `model.save()` are using
the up-to-date `.keras` format, and you're done.

If your `tf.keras` model does not include custom components, you can start running it on top of JAX or PyTorch immediately.

If it does include custom components (e.g. custom layers or a custom `train_step()`), it is usually possible to convert it
to a backend-agnostic implementation in just a few minutes.

In addition, Keras models can consume datasets in any format, regardless of the backend you're using:
you can train your models with your existing `tf.data.Dataset` pipelines or PyTorch `DataLoaders`.

## Why use Keras Core?

- Run your high-level Keras workflows on top of any framework -- benefiting at will from the advantages of each framework,
e.g. the scalability and performance of JAX or the production ecosystem options of TensorFlow.
- Write custom components (e.g. layers, models, metrics) that you can use in low-level workflows in any framework.
    - You can take a Keras model and train it in a training loop written from scratch in native TF, JAX, or PyTorch.
    - You can take a Keras model and use it as part of a PyTorch-native `Module` or as part of a JAX-native model function.
- Make your ML code future-proof by avoiding framework lock-in.
- As a PyTorch user: get access to power and usability of Keras, at last!
- As a JAX user: get access to a fully-featured, battle-tested, well-documented modeling and training library.
