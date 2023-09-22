import random

import numpy as np

from keras import backend
from keras.api_export import keras_export
from keras.utils.module_utils import tensorflow as tf


@keras_export("keras.utils.set_random_seed")
def set_random_seed(seed):
    """Sets all random seeds (Python, NumPy, and backend framework, e.g. TF).

    You can use this utility to make almost any Keras program fully
    deterministic. Some limitations apply in cases where network communications
    are involved (e.g. parameter server distribution), which creates additional
    sources of randomness, or when certain non-deterministic cuDNN ops are
    involved.

    Calling this utility is equivalent to the following:

    ```python
    import random
    import numpy as np
    from keras.utils.module_utils import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    ```

    Note that the TensorFlow seed is set even if you're not using TensorFlow
    as your backend framework, since many workflows leverage `tf.data`
    pipelines (which feature random shuffling). Likewise many workflows
    might leverage NumPy APIs.

    Arguments:
        seed: Integer, the random seed to use.
    """
    if not isinstance(seed, int):
        raise ValueError(
            "Expected `seed` argument to be an integer. "
            f"Received: seed={seed} (of type {type(seed)})"
        )
    random.seed(seed)
    np.random.seed(seed)
    if tf.available:
        tf.random.set_seed(seed)
    if backend.backend() == "torch":
        import torch

        torch.manual_seed(seed)
