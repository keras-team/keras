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

# Imports needed for deserialization.

import platform

import tensorflow.compat.v2 as tf
from absl import logging

from keras import backend
from keras.optimizers import adadelta
from keras.optimizers import adafactor
from keras.optimizers import adagrad
from keras.optimizers import adam
from keras.optimizers import adamax
from keras.optimizers import adamw
from keras.optimizers import ftrl
from keras.optimizers import nadam
from keras.optimizers import optimizer as base_optimizer
from keras.optimizers import rmsprop
from keras.optimizers import sgd
from keras.optimizers.legacy import adadelta as adadelta_legacy
from keras.optimizers.legacy import adagrad as adagrad_legacy
from keras.optimizers.legacy import adam as adam_legacy
from keras.optimizers.legacy import adamax as adamax_legacy
from keras.optimizers.legacy import ftrl as ftrl_legacy
from keras.optimizers.legacy import gradient_descent as gradient_descent_legacy
from keras.optimizers.legacy import nadam as nadam_legacy
from keras.optimizers.legacy import optimizer_v2 as base_optimizer_legacy
from keras.optimizers.legacy import rmsprop as rmsprop_legacy
from keras.optimizers.legacy.adadelta import Adadelta
from keras.optimizers.legacy.adagrad import Adagrad
from keras.optimizers.legacy.adam import Adam
from keras.optimizers.legacy.adamax import Adamax
from keras.optimizers.legacy.ftrl import Ftrl

# Symbols to be accessed under keras.optimizers. To be replaced with
# optimizers v2022 when they graduate out of experimental.
from keras.optimizers.legacy.gradient_descent import SGD
from keras.optimizers.legacy.nadam import Nadam
from keras.optimizers.legacy.rmsprop import RMSprop
from keras.optimizers.optimizer_v1 import Optimizer
from keras.optimizers.optimizer_v1 import TFOptimizer
from keras.optimizers.schedules import learning_rate_schedule
from keras.saving.legacy import serialization as legacy_serialization
from keras.saving.legacy.serialization import deserialize_keras_object
from keras.saving.legacy.serialization import serialize_keras_object

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.optimizers.serialize")
def serialize(optimizer, use_legacy_format=False):
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
    if use_legacy_format:
        return legacy_serialization.serialize_keras_object(optimizer)
    return serialize_keras_object(optimizer)


def is_arm_mac():
    return platform.system() == "Darwin" and platform.processor() == "arm"


@keras_export("keras.optimizers.deserialize")
def deserialize(config, custom_objects=None, use_legacy_format=False, **kwargs):
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
    if kwargs:
        raise TypeError(f"Invalid keyword arguments: {kwargs}")
    if len(config["config"]) > 0:
        # If the optimizer config is not empty, then we use the value of
        # `is_legacy_optimizer` to override `use_legacy_optimizer`. If
        # `is_legacy_optimizer` does not exist in config, it means we are
        # using the legacy optimzier.
        use_legacy_optimizer = config["config"].get("is_legacy_optimizer", True)
    if (
        tf.__internal__.tf2.enabled()
        and tf.executing_eagerly()
        and not is_arm_mac()
        and not use_legacy_optimizer
    ):
        # We observed a slowdown of optimizer on M1 Mac, so we fall back to the
        # legacy optimizer for M1 users now, see b/263339144 for more context.
        all_classes = {
            "adadelta": adadelta.Adadelta,
            "adagrad": adagrad.Adagrad,
            "adam": adam.Adam,
            "adamax": adamax.Adamax,
            "experimentaladadelta": adadelta.Adadelta,
            "experimentaladagrad": adagrad.Adagrad,
            "experimentaladam": adam.Adam,
            "experimentalsgd": sgd.SGD,
            "nadam": nadam.Nadam,
            "rmsprop": rmsprop.RMSprop,
            "sgd": sgd.SGD,
            "ftrl": ftrl.Ftrl,
            "lossscaleoptimizer": loss_scale_optimizer.LossScaleOptimizerV3,
            "lossscaleoptimizerv3": loss_scale_optimizer.LossScaleOptimizerV3,
            # LossScaleOptimizerV1 was an old version of LSO that was removed.
            # Deserializing it turns it into a LossScaleOptimizer
            "lossscaleoptimizerv1": loss_scale_optimizer.LossScaleOptimizer,
        }
    else:
        all_classes = {
            "adadelta": adadelta_legacy.Adadelta,
            "adagrad": adagrad_legacy.Adagrad,
            "adam": adam_legacy.Adam,
            "adamax": adamax_legacy.Adamax,
            "experimentaladadelta": adadelta.Adadelta,
            "experimentaladagrad": adagrad.Adagrad,
            "experimentaladam": adam.Adam,
            "experimentalsgd": sgd.SGD,
            "nadam": nadam_legacy.Nadam,
            "rmsprop": rmsprop_legacy.RMSprop,
            "sgd": gradient_descent_legacy.SGD,
            "ftrl": ftrl_legacy.Ftrl,
            "lossscaleoptimizer": loss_scale_optimizer.LossScaleOptimizer,
            "lossscaleoptimizerv3": loss_scale_optimizer.LossScaleOptimizerV3,
            # LossScaleOptimizerV1 was an old version of LSO that was removed.
            # Deserializing it turns it into a LossScaleOptimizer
            "lossscaleoptimizerv1": loss_scale_optimizer.LossScaleOptimizer,
        }

    # Make deserialization case-insensitive for built-in optimizers.
    if config["class_name"].lower() in all_classes:
        config["class_name"] = config["class_name"].lower()

    if use_legacy_format:
        return legacy_serialization.deserialize_keras_object(
            config,
            module_objects=all_classes,
            custom_objects=custom_objects,
            printable_module_name="optimizer",
        )

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
    # loss_scale_optimizer has a direct dependency of optimizer, import here
    # rather than top to avoid the cyclic dependency.
    from keras.mixed_precision import (
        loss_scale_optimizer,
    )

    if not isinstance(optimizer, base_optimizer.Optimizer):
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

    if isinstance(optimizer, loss_scale_optimizer.LossScaleOptimizerV3):
        # For LossScaleOptimizers, recursively convert the inner optimizer
        config["inner_optimizer"] = convert_to_legacy_optimizer(
            optimizer.inner_optimizer
        )
        if optimizer_name == "lossscaleoptimizerv3":
            optimizer_name = "lossscaleoptimizer"

    # Learning rate can be a custom LearningRateSchedule, which is stored as
    # a dict in config, and cannot be deserialized.
    if hasattr(optimizer, "_learning_rate") and isinstance(
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
        identifier: Optimizer identifier, one of - String: name of an optimizer
          - Dictionary: configuration dictionary. - Keras Optimizer instance (it
          will be returned unchanged). - TensorFlow Optimizer instance (it will
          be wrapped as a Keras Optimizer).

    Returns:
        A Keras Optimizer instance.

    Raises:
        ValueError: If `identifier` cannot be interpreted.
    """
    use_legacy_optimizer = kwargs.pop("use_legacy_optimizer", False)
    if kwargs:
        raise TypeError(f"Invalid keyword arguments: {kwargs}")
    if isinstance(
        identifier,
        (
            Optimizer,
            base_optimizer_legacy.OptimizerV2,
        ),
    ):
        return identifier
    elif isinstance(identifier, base_optimizer.Optimizer):
        if tf.__internal__.tf2.enabled() and not is_arm_mac():
            return identifier
        else:
            # If TF2 is disabled or on a M1 mac, we convert to the legacy
            # optimizer. We observed a slowdown of optimizer on M1 Mac, so we
            # fall back to the legacy optimizer for now, see b/263339144
            # for more context.
            optimizer_name = identifier.__class__.__name__
            logging.warning(
                "There is a known slowdown when using v2.11+ Keras optimizers "
                "on M1/M2 Macs. Falling back to the "
                "legacy Keras optimizer, i.e., "
                f"`tf.keras.optimizers.legacy.{optimizer_name}`."
            )
            return convert_to_legacy_optimizer(identifier)

    # Wrap legacy TF optimizer instances
    elif isinstance(identifier, tf.compat.v1.train.Optimizer):
        opt = TFOptimizer(identifier)
        backend.track_tf_optimizer(opt)
        return opt
    elif isinstance(identifier, dict):
        use_legacy_format = "module" not in identifier
        return deserialize(
            identifier,
            use_legacy_optimizer=use_legacy_optimizer,
            use_legacy_format=use_legacy_format,
        )
    elif isinstance(identifier, str):
        config = {"class_name": str(identifier), "config": {}}
        return deserialize(
            config,
            use_legacy_optimizer=use_legacy_optimizer,
        )
    else:
        raise ValueError(
            f"Could not interpret optimizer identifier: {identifier}"
        )
