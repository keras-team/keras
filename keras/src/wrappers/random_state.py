import os
import random
from contextlib import contextmanager
from typing import Generator, Iterator

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.python.eager import context
    from tensorflow.python.framework import config, ops

    def tf_set_seed(seed: int) -> None:
        tf.random.set_seed(seed)

    def tf_get_seed() -> Iterator[int]:
        if context.executing_eagerly():
            return context.global_seed()
        else:
            return ops.get_default_graph().seed

    def tf_enable_op_determinism() -> bool:
        was_enabled = config.is_op_determinism_enabled()
        config.enable_op_determinism()
        return was_enabled

    def tf_disable_op_determinism() -> None:
        config.disable_op_determinism()

except ImportError:

    def tf_set_seed(seed: int) -> None:
        pass

    def tf_get_seed() -> int:
        return 0

    def tf_enable_op_determinism() -> bool:
        return False

    def tf_disable_op_determinism() -> None:
        return None


@contextmanager
def tensorflow_random_state(seed: int) -> Generator[None, None, None]:
    # Save values
    origin_gpu_det = os.environ.get("TF_DETERMINISTIC_OPS", None)
    orig_random_state = random.getstate()
    orig_np_random_state = np.random.get_state()
    tf_random_seed = tf_get_seed()
    determinism_enabled = None
    try:
        # Set values
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        random.seed(seed)
        np.random.seed(seed)
        tf_set_seed(seed)
        determinism_enabled = tf_enable_op_determinism()
        yield
    finally:
        # Reset values
        if origin_gpu_det is not None:
            os.environ["TF_DETERMINISTIC_OPS"] = origin_gpu_det
        else:
            os.environ.pop("TF_DETERMINISTIC_OPS")
        random.setstate(orig_random_state)
        np.random.set_state(orig_np_random_state)
        tf_set_seed(tf_random_seed)
        if determinism_enabled is False:
            tf_disable_op_determinism()
