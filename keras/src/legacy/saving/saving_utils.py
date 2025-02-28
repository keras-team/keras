import json
import threading

from absl import logging

from keras.src import backend
from keras.src import layers
from keras.src import losses
from keras.src import metrics as metrics_module
from keras.src import models
from keras.src import optimizers
from keras.src import tree
from keras.src.legacy.saving import serialization
from keras.src.saving import object_registration

MODULE_OBJECTS = threading.local()

# Legacy lambda arguments not found in Keras 3
LAMBDA_DEP_ARGS = (
    "module",
    "function_type",
    "output_shape_type",
    "output_shape_module",
)


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

    batch_input_shape = config["config"].pop("batch_input_shape", None)
    if batch_input_shape is not None:
        if config["class_name"] == "InputLayer":
            config["config"]["batch_shape"] = batch_input_shape
        else:
            config["config"]["input_shape"] = batch_input_shape

    axis = config["config"].pop("axis", None)
    if axis is not None:
        if isinstance(axis, list) and len(axis) == 1:
            config["config"]["axis"] = int(axis[0])
        elif isinstance(axis, (int, float)):
            config["config"]["axis"] = int(axis)

    # Handle backwards compatibility for Keras lambdas
    if config["class_name"] == "Lambda":
        for dep_arg in LAMBDA_DEP_ARGS:
            _ = config["config"].pop(dep_arg, None)
        function_config = config["config"]["function"]
        if isinstance(function_config, list):
            function_dict = {"class_name": "__lambda__", "config": {}}
            function_dict["config"]["code"] = function_config[0]
            function_dict["config"]["defaults"] = function_config[1]
            function_dict["config"]["closure"] = function_config[2]
            config["config"]["function"] = function_dict

    # TODO(nkovela): Swap find and replace args during Keras 3.0 release
    # Replace keras refs with keras
    config = _find_replace_nested_dict(config, "keras.", "keras.")

    return serialization.deserialize_keras_object(
        config,
        module_objects=MODULE_OBJECTS.ALL_OBJECTS,
        custom_objects=custom_objects,
        printable_module_name="layer",
    )


def model_metadata(model, include_optimizer=True, require_config=True):
    """Returns a dictionary containing the model metadata."""
    from keras.src import __version__ as keras_version

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
        # Ensure backwards compatibility for optimizers in legacy H5 files
        optimizer = _resolve_compile_arguments_compat(
            optimizer, optimizer_config, optimizers
        )

        # Recover losses.
        loss = None
        loss_config = training_config.get("loss", None)
        if loss_config is not None:
            loss = _deserialize_nested_config(losses.deserialize, loss_config)
            # Ensure backwards compatibility for losses in legacy H5 files
            loss = _resolve_compile_arguments_compat(loss, loss_config, losses)

        # Recover metrics.
        metrics = None
        metrics_config = training_config.get("metrics", None)
        if metrics_config is not None:
            metrics = _deserialize_nested_config(
                _deserialize_metric, metrics_config
            )
            # Ensure backwards compatibility for metrics in legacy H5 files
            metrics = _resolve_compile_arguments_compat(
                metrics, metrics_config, metrics_module
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
    return metrics_module.deserialize(metric_config)


def _find_replace_nested_dict(config, find, replace):
    dict_str = json.dumps(config)
    dict_str = dict_str.replace(find, replace)
    config = json.loads(dict_str)
    return config


def _resolve_compile_arguments_compat(obj, obj_config, module):
    """Resolves backwards compatibility issues with training config arguments.

    This helper function accepts built-in Keras modules such as optimizers,
    losses, and metrics to ensure an object being deserialized is compatible
    with Keras 3 built-ins. For legacy H5 files saved within Keras 3,
    this does nothing.
    """
    if isinstance(obj, str) and obj not in module.ALL_OBJECTS_DICT:
        obj = module.get(obj_config["config"]["name"])
    return obj


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
