# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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


"""Built-in optimizer classes.

For more examples see the base class `tf.keras.optimizers.Optimizer`.
"""

import tensorflow.compat.v2 as tf

# Imports needed for deserialization.
from keras import backend
from keras.optimizers.legacy import adadelta as adadelta_legacy
from keras.optimizers.legacy import adagrad as adagrad_legacy
from keras.optimizers.legacy import adam as adam_legacy
from keras.optimizers.legacy import adamax as adamax_legacy
from keras.optimizers.legacy import ftrl as ftrl_legacy
from keras.optimizers.legacy import nadam as nadam_legacy
from keras.optimizers.legacy import optimizer as optimizer_legacy
from keras.optimizers.legacy import rmsprop as rmsprop_legacy
from keras.optimizers.legacy import sgd as sgd_legacy
from keras.optimizers.optimizer_experimental import (
    adadelta as adadelta_experimental,
)
from keras.optimizers.optimizer_experimental import adafactor
from keras.optimizers.optimizer_experimental import (
    adagrad as adagrad_experimental,
)
from keras.optimizers.optimizer_experimental import adam as adam_experimental
from keras.optimizers.optimizer_experimental import (
    adamax as adamax_experimental,
)
from keras.optimizers.optimizer_experimental import adamw as adamw_experimental
from keras.optimizers.optimizer_experimental import ftrl as ftrl_experimental
from keras.optimizers.optimizer_experimental import nadam as nadam_experimental
from keras.optimizers.optimizer_experimental import (
    optimizer as optimizer_experimental,
)
from keras.optimizers.optimizer_experimental import (
    rmsprop as rmsprop_experimental,
)
from keras.optimizers.optimizer_experimental import sgd as sgd_experimental
from keras.optimizers.optimizer_v1 import Optimizer
from keras.optimizers.optimizer_v1 import TFOptimizer
from keras.optimizers.optimizer_v2 import adadelta as adadelta_v2
from keras.optimizers.optimizer_v2 import adagrad as adagrad_v2
from keras.optimizers.optimizer_v2 import adam as adam_v2
from keras.optimizers.optimizer_v2 import adamax as adamax_v2
from keras.optimizers.optimizer_v2 import ftrl
from keras.optimizers.optimizer_v2 import (
    gradient_descent as gradient_descent_v2,
)
from keras.optimizers.optimizer_v2 import nadam as nadam_v2
from keras.optimizers.optimizer_v2 import optimizer_v2 as base_optimizer_v2
from keras.optimizers.optimizer_v2 import rmsprop as rmsprop_v2
from keras.optimizers.optimizer_v2.adadelta import Adadelta
from keras.optimizers.optimizer_v2.adagrad import Adagrad
from keras.optimizers.optimizer_v2.adam import Adam
from keras.optimizers.optimizer_v2.adamax import Adamax
from keras.optimizers.optimizer_v2.ftrl import Ftrl

# Symbols to be accessed under keras.optimizers. To be replaced with
# optimizers v2022 when they graduate out of experimental.
from keras.optimizers.optimizer_v2.gradient_descent import SGD
from keras.optimizers.optimizer_v2.nadam import Nadam
from keras.optimizers.optimizer_v2.rmsprop import RMSprop
from keras.optimizers.schedules import learning_rate_schedule
from keras.saving.legacy.serialization import deserialize_keras_object
from keras.saving.legacy.serialization import serialize_keras_object

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.optimizers.serialize")
def serialize(optimizer):
    """Serialize the optimizer configuration to JSON compatible python dict.

    The configuration can be used for persistence and reconstruct the
    `Optimizer` instance again.

    >>> tf.keras.optimizers.serialize(tf.keras.optimizers.legacy.SGD())
    {'class_name': 'SGD', 'config': {'name': 'SGD', 'learning_rate': 0.01,
                                     'decay': 0.0, 'momentum': 0.0,
                                     'nesterov': False}}

    Args:
      optimizer: An `Optimizer` instance to serialize.

    Returns:
      Python dict which contains the configuration of the input optimizer.
    """
    return serialize_keras_object(optimizer)


@keras_export("keras.optimizers.deserialize")
def deserialize(config, custom_objects=None, **kwargs):
    """Inverse of the `serialize` function.

    Args:
        config: Optimizer configuration dictionary.
        custom_objects: Optional dictionary mapping names (strings) to custom
          objects (classes and functions) to be considered during
          deserialization.

    Returns:
        A Keras Optimizer instance.
    """
    # loss_scale_optimizer has a direct dependency of optimizer, import here
    # rather than top to avoid the cyclic dependency.
    from keras.mixed_precision import (
        loss_scale_optimizer,
    )

    use_legacy_optimizer = kwargs.pop("use_legacy_optimizer", False)
    if len(config["config"]) > 0:
        # If the optimizer config is not empty, then we use the value of
        # `is_legacy_optimizer` to override `use_legacy_optimizer`. If
        # `is_legacy_optimizer` does not exist in config, it means we are
        # using the legacy optimzier.
        use_legacy_optimizer = config["config"].get("is_legacy_optimizer", True)
    if (
        tf.__internal__.tf2.enabled()
        and tf.executing_eagerly()
        and not use_legacy_optimizer
    ):
        all_classes = {
            "adadelta": adadelta_experimental.Adadelta,
            "adagrad": adagrad_experimental.Adagrad,
            "adam": adam_experimental.Adam,
            "adamax": adamax_experimental.Adamax,
            "experimentaladadelta": adadelta_experimental.Adadelta,
            "experimentaladagrad": adagrad_experimental.Adagrad,
            "experimentaladam": adam_experimental.Adam,
            "experimentalsgd": sgd_experimental.SGD,
            "nadam": nadam_experimental.Nadam,
            "rmsprop": rmsprop_experimental.RMSprop,
            "sgd": sgd_experimental.SGD,
            "ftrl": ftrl_experimental.Ftrl,
            "lossscaleoptimizer": loss_scale_optimizer.LossScaleOptimizerV3,
            "lossscaleoptimizerv3": loss_scale_optimizer.LossScaleOptimizerV3,
            # LossScaleOptimizerV1 was an old version of LSO that was removed.
            # Deserializing it turns it into a LossScaleOptimizer
            "lossscaleoptimizerv1": loss_scale_optimizer.LossScaleOptimizer,
        }
    else:
        all_classes = {
            "adadelta": adadelta_v2.Adadelta,
            "adagrad": adagrad_v2.Adagrad,
            "adam": adam_v2.Adam,
            "adamax": adamax_v2.Adamax,
            "experimentaladadelta": adadelta_experimental.Adadelta,
            "experimentaladagrad": adagrad_experimental.Adagrad,
            "experimentaladam": adam_experimental.Adam,
            "experimentalsgd": sgd_experimental.SGD,
            "nadam": nadam_v2.Nadam,
            "rmsprop": rmsprop_v2.RMSprop,
            "sgd": gradient_descent_v2.SGD,
            "ftrl": ftrl.Ftrl,
            "lossscaleoptimizer": loss_scale_optimizer.LossScaleOptimizer,
            "lossscaleoptimizerv3": loss_scale_optimizer.LossScaleOptimizerV3,
            # LossScaleOptimizerV1 was an old version of LSO that was removed.
            # Deserializing it turns it into a LossScaleOptimizer
            "lossscaleoptimizerv1": loss_scale_optimizer.LossScaleOptimizer,
        }

    # Make deserialization case-insensitive for built-in optimizers.
    if config["class_name"].lower() in all_classes:
        config["class_name"] = config["class_name"].lower()
    return deserialize_keras_object(
        config,
        module_objects=all_classes,
        custom_objects=custom_objects,
        printable_module_name="optimizer",
    )


@keras_export(
    "keras.__internal__.optimizers.convert_to_legacy_optimizer", v1=[]
)
def convert_to_legacy_optimizer(optimizer):
    """Convert experimental optimizer to legacy optimizer.

    This function takes in a `tf.keras.optimizers.experimental.Optimizer`
    instance and converts it to the corresponding
    `tf.keras.optimizers.legacy.Optimizer` instance.
    For example, `tf.keras.optimizers.experimental.Adam(...)` to
    `tf.keras.optimizers.legacy.Adam(...)`.

    Args:
        optimizer: An instance of `tf.keras.optimizers.experimental.Optimizer`.
    """
    if not isinstance(optimizer, optimizer_experimental.Optimizer):
        raise ValueError(
            "`convert_to_legacy_optimizer` should only be called "
            "on instances of `tf.keras.optimizers.Optimizer`, but "
            f"received {optimizer} of type {type(optimizer)}."
        )
    optimizer_name = optimizer.__class__.__name__.lower()
    config = optimizer.get_config()
    # Remove fields that only exist in experimental optimizer.
    keys_to_remove = [
        "weight_decay",
        "use_ema",
        "ema_momentum",
        "ema_overwrite_frequency",
        "jit_compile",
        "is_legacy_optimizer",
    ]
    for key in keys_to_remove:
        config.pop(key, None)
    # Learning rate can be a custom LearningRateSchedule, which is stored as
    # a dict in config, and cannot be deserialized.
    if isinstance(
        optimizer._learning_rate, learning_rate_schedule.LearningRateSchedule
    ):
        config["learning_rate"] = optimizer._learning_rate
    legacy_optimizer_config = {
        "class_name": optimizer_name,
        "config": config,
    }
    return deserialize(legacy_optimizer_config, use_legacy_optimizer=True)


@keras_export("keras.optimizers.get")
def get(identifier, **kwargs):
    """Retrieves a Keras Optimizer instance.

    Args:
        identifier: Optimizer identifier, one of
            - String: name of an optimizer
            - Dictionary: configuration dictionary.
            - Keras Optimizer instance (it will be returned unchanged).
            - TensorFlow Optimizer instance (it will be wrapped as a Keras
              Optimizer).

    Returns:
        A Keras Optimizer instance.

    Raises:
        ValueError: If `identifier` cannot be interpreted.
    """
    use_legacy_optimizer = kwargs.pop("use_legacy_optimizer", False)
    if isinstance(
        identifier,
        (
            Optimizer,
            base_optimizer_v2.OptimizerV2,
        ),
    ):
        return identifier
    elif isinstance(identifier, optimizer_experimental.Optimizer):
        if tf.__internal__.tf2.enabled():
            return identifier
        else:
            # If TF2 is disabled, we convert to the legacy optimizer.
            return convert_to_legacy_optimizer(identifier)

    # Wrap legacy TF optimizer instances
    elif isinstance(identifier, tf.compat.v1.train.Optimizer):
        opt = TFOptimizer(identifier)
        backend.track_tf_optimizer(opt)
        return opt
    elif isinstance(identifier, dict):
        return deserialize(
            identifier, use_legacy_optimizer=use_legacy_optimizer
        )
    elif isinstance(identifier, str):
        config = {"class_name": str(identifier), "config": {}}
        return deserialize(config, use_legacy_optimizer=use_legacy_optimizer)
    else:
        raise ValueError(
            f"Could not interpret optimizer identifier: {identifier}"
        )
