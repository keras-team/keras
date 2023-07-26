import threading

import tree
from absl import logging

from keras_core import backend
from keras_core import layers
from keras_core import losses
from keras_core import metrics
from keras_core import models
from keras_core import optimizers
from keras_core.legacy.saving import serialization
from keras_core.saving import object_registration

MODULE_OBJECTS = threading.local()


def model_from_config(config, custom_objects=None):
    """Instantiates a Keras model from its config.

    Args:
        config: Configuration dictionary.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        A Keras model instance (uncompiled).

    Raises:
        TypeError: if `config` is not a dictionary.
    """
    if isinstance(config, list):
        raise TypeError(
            "`model_from_config` expects a dictionary, not a list. "
            f"Received: config={config}. Did you meant to use "
            "`Sequential.from_config(config)`?"
        )

    global MODULE_OBJECTS

    if not hasattr(MODULE_OBJECTS, "ALL_OBJECTS"):
        MODULE_OBJECTS.ALL_OBJECTS = layers.__dict__
        MODULE_OBJECTS.ALL_OBJECTS["InputLayer"] = layers.InputLayer
        MODULE_OBJECTS.ALL_OBJECTS["Functional"] = models.Functional
        MODULE_OBJECTS.ALL_OBJECTS["Model"] = models.Model
        MODULE_OBJECTS.ALL_OBJECTS["Sequential"] = models.Sequential

    return serialization.deserialize_keras_object(
        config,
        module_objects=MODULE_OBJECTS.ALL_OBJECTS,
        custom_objects=custom_objects,
        printable_module_name="layer",
    )


def model_metadata(model, include_optimizer=True, require_config=True):
    """Returns a dictionary containing the model metadata."""
    from keras_core import __version__ as keras_version

    model_config = {"class_name": model.__class__.__name__}
    try:
        model_config["config"] = model.get_config()
    except NotImplementedError as e:
        if require_config:
            raise e

    metadata = dict(
        keras_version=str(keras_version),
        backend=backend.backend(),
        model_config=model_config,
    )
    if getattr(model, "optimizer", False) and include_optimizer:
        if model.compiled:
            training_config = model._compile_config.config
            training_config.pop("optimizer", None)  # Handled separately.
            metadata["training_config"] = _serialize_nested_config(
                training_config
            )
            optimizer_config = {
                "class_name": object_registration.get_registered_name(
                    model.optimizer.__class__
                ),
                "config": model.optimizer.get_config(),
            }
            metadata["training_config"]["optimizer_config"] = optimizer_config
    return metadata


def compile_args_from_training_config(training_config, custom_objects=None):
    """Return model.compile arguments from training config."""
    if custom_objects is None:
        custom_objects = {}

    with object_registration.CustomObjectScope(custom_objects):
        optimizer_config = training_config["optimizer_config"]
        optimizer = optimizers.deserialize(optimizer_config)

        # Recover losses.
        loss = None
        loss_config = training_config.get("loss", None)
        if loss_config is not None:
            loss = _deserialize_nested_config(losses.deserialize, loss_config)

        # Recover metrics.
        metrics = None
        metrics_config = training_config.get("metrics", None)
        if metrics_config is not None:
            metrics = _deserialize_nested_config(
                _deserialize_metric, metrics_config
            )

        # Recover weighted metrics.
        weighted_metrics = None
        weighted_metrics_config = training_config.get("weighted_metrics", None)
        if weighted_metrics_config is not None:
            weighted_metrics = _deserialize_nested_config(
                _deserialize_metric, weighted_metrics_config
            )

        loss_weights = training_config["loss_weights"]

    return dict(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
        loss_weights=loss_weights,
    )


def _serialize_nested_config(config):
    """Serialized a nested structure of Keras objects."""

    def _serialize_fn(obj):
        if callable(obj):
            return serialization.serialize_keras_object(obj)
        return obj

    return tree.map_structure(_serialize_fn, config)


def _deserialize_nested_config(deserialize_fn, config):
    """Deserializes arbitrary Keras `config` using `deserialize_fn`."""

    def _is_single_object(obj):
        if isinstance(obj, dict) and "class_name" in obj:
            return True  # Serialized Keras object.
        if isinstance(obj, str):
            return True  # Serialized function or string.
        return False

    if config is None:
        return None
    if _is_single_object(config):
        return deserialize_fn(config)
    elif isinstance(config, dict):
        return {
            k: _deserialize_nested_config(deserialize_fn, v)
            for k, v in config.items()
        }
    elif isinstance(config, (tuple, list)):
        return [
            _deserialize_nested_config(deserialize_fn, obj) for obj in config
        ]

    raise ValueError(
        "Saved configuration not understood. Configuration should be a "
        f"dictionary, string, tuple or list. Received: config={config}."
    )


def _deserialize_metric(metric_config):
    """Deserialize metrics, leaving special strings untouched."""
    if metric_config in ["accuracy", "acc", "crossentropy", "ce"]:
        # Do not deserialize accuracy and cross-entropy strings as we have
        # special case handling for these in compile, based on model output
        # shape.
        return metric_config
    return metrics.deserialize(metric_config)


def try_build_compiled_arguments(model):
    try:
        if not model.compiled_loss.built:
            model.compiled_loss.build(model.outputs)
        if not model.compiled_metrics.built:
            model.compiled_metrics.build(model.outputs, model.outputs)
    except:
        logging.warning(
            "Compiled the loaded model, but the compiled metrics have "
            "yet to be built. `model.compile_metrics` will be empty "
            "until you train or evaluate the model."
        )
