# Keras 3: A new multi-backend Keras

Keras 3 is a new multi-backend implementation of the Keras API, with support for TensorFlow, JAX, and PyTorch.

## Installation

### Install with pip

Keras 3 is available as a preview release on PyPI named `keras-core`.
Keras 2 (`tf.keras`) is distributed along with the `tensorflow` package.

1. Install `keras-core`:

```
pip install keras-core
```

2. Install backend package(s).

To use `keras-core`, you should also install the backend of choice: `tensorflow`, `jax`, or `torch`.
Note that `tensorflow` is required for using certain Keras 3 features: certain preprocessing layers
as well as `tf.data` pipelines.

**Note:** If you are using the `keras-core` package you also need to switch your Keras import.
Use `import keras_core as keras`. This is a temporary step until the release of Keras 3 on PyPI.

### Local installation

Keras 3 is compatible with Linux and MacOS systems. To install a local development version:

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run installation command from the root directory.

```
python pip_build.py --install
```

The `requirements.txt` file will install a CPU-only version of TensorFlow, JAX, and PyTorch. For GPU support, we also
provide a separate `requirements-{backend}-cuda.txt` for TensorFlow, JAX, and PyTorch. These install all CUDA
dependencies via `pip` and expect a NVIDIA driver to be pre-installed. We recommend a clean python environment for each
backend to avoid CUDA version mismatches. As an example, here is how to create a Jax GPU environment with `conda`:

```shell
conda create -y -n keras-jax python=3.10
conda activate keras-jax
pip install -r requirements-jax-cuda.txt
python pip_build.py --install
```

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

**Note:** The backend must be configured before importing `keras`, and the backend cannot be changed after 
the package has been imported.

## Backwards compatibility

Keras 3 is intended to work as a drop-in replacement for `tf.keras` (when using the TensorFlow backend). Just take your
existing `tf.keras` code, make sure that your calls to `model.save()` are using the up-to-date `.keras` format, and you're
done.

If your `tf.keras` model does not include custom components, you can start running it on top of JAX or PyTorch immediately.

If it does include custom components (e.g. custom layers or a custom `train_step()`), it is usually possible to convert it
to a backend-agnostic implementation in just a few minutes.

In addition, Keras models can consume datasets in any format, regardless of the backend you're using:
you can train your models with your existing `tf.data.Dataset` pipelines or PyTorch `DataLoaders`.

## Keras 3 timeline

At the moment, we are releasing Keras 3 as a preview release with under the `keras-core` name on PyPI. We encourage anyone
interested in the future of the library to try it out and give feedback.

You can find the current stable release of Keras 2 at the [tf-keras](https://github.com/keras-team/tf-keras) repository.

We will share updates on the release timeline as soon as they are available.

## Why use Keras 3?

- Run your high-level Keras workflows on top of any framework -- benefiting at will from the advantages of each framework,
e.g. the scalability and performance of JAX or the production ecosystem options of TensorFlow.
- Write custom components (e.g. layers, models, metrics) that you can use in low-level workflows in any framework.
    - You can take a Keras model and train it in a training loop written from scratch in native TF, JAX, or PyTorch.
    - You can take a Keras model and use it as part of a PyTorch-native `Module` or as part of a JAX-native model function.
- Make your ML code future-proof by avoiding framework lock-in.
- As a PyTorch user: get access to power and usability of Keras, at last!
- As a JAX user: get access to a fully-featured, battle-tested, well-documented modeling and training library.
