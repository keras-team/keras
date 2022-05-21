# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities related to distributed training."""

from absl import flags
from keras import backend

import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS


# TODO(b/118776054): Currently we support global batch size for TPUStrategy and
# core MirroredStrategy only. Remove this check when contrib MirroredStrategy is
# no longer needed.
def global_batch_size_supported(distribution_strategy):
    return (
        distribution_strategy.extended._global_batch_size
    )  # pylint: disable=protected-access


def call_replica_local_fn(fn, *args, **kwargs):
    """Call a function that uses replica-local variables.

    This function correctly handles calling `fn` in a cross-replica
    context.

    Args:
      fn: The function to call.
      *args: Positional arguments to the `fn`.
      **kwargs: Keyword argument to `fn`.

    Returns:
      The result of calling `fn`.
    """
    # TODO(b/132666209): Remove this function when we support assign_*
    # for replica-local variables.
    strategy = None
    if "strategy" in kwargs:
        strategy = kwargs.pop("strategy")
    else:
        if tf.distribute.has_strategy():
            strategy = tf.distribute.get_strategy()

    # TODO(b/120571621): TPUStrategy does not implement replica-local variables.
    is_tpu = backend.is_tpu_strategy(strategy)
    if (not is_tpu) and strategy and tf.distribute.in_cross_replica_context():
        with strategy.scope():
            return strategy.extended.call_for_each_replica(fn, args, kwargs)
    return fn(*args, **kwargs)


def is_distributed_variable(v):
    """Returns whether `v` is a distributed variable."""
    return isinstance(v, tf.distribute.DistributedValues) and isinstance(
        v, tf.Variable
    )


def get_strategy():
    """Creates a `tf.distribute.Strategy` object from flags.

    Example usage:

    ```python
    strategy = utils.get_strategy()
    with strategy.scope():
      model = tf.keras.Sequential([tf.keras.layers.Dense(10)])

    model.compile(...)
    train_ds, test_ds = ...
    model.fit(train_ds, validation_data=test_ds, epochs=10)
    ```

    Returns:
      `tf.distribute.Strategy` instance.
    """
    cls = FLAGS.keras_distribute_strategy_class
    accepted_strats = {
        "tpu",
        "multi_worker_mirrored",
        "mirrored",
        "parameter_server",
        "one_device",
    }
    if cls == "tpu":
        tpu_addr = FLAGS.keras_distribute_strategy_tpu_addr
        if not tpu_addr:
            raise ValueError(
                "When using a TPU strategy, you must set the flag "
                "`keras_distribute_strategy_tpu_addr` (TPU address)."
            )
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_addr
        )
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    elif cls == "multi_worker_mirrored":
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    elif cls == "mirrored":
        strategy = tf.distribute.MirroredStrategy()
    elif cls == "parameter_server":
        cluster_resolver = (
            tf.distribute.cluster_resolver.TFConfigClusterResolver()
        )
        strategy = tf.distribute.experimental.ParameterServerStrategy(
            cluster_resolver
        )
    elif cls == "one_device":
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    else:
        raise ValueError(
            "Unknown distribution strategy flag. Received: "
            f"keras_distribute_strategy_class={cls}. "
            f"It should be one of {accepted_strats}"
        )
    return strategy
