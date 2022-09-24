# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Python-based idempotent model-saving functionality."""

import datetime
import json
import os
import tempfile
import threading
import uuid
import warnings
import zipfile

import tensorflow.compat.v2 as tf

import keras
from keras import losses
from keras.engine import base_layer
from keras.optimizers.optimizer_experimental import optimizer
from keras.saving.experimental.serialization_lib import deserialize_keras_object
from keras.saving.experimental.serialization_lib import serialize_keras_object
from keras.utils import io_utils

# isort: off

_SELF_DIRNAME = "self"
_CONFIG_FILENAME = "config.json"
_METADATA_FILENAME = "metadata.json"
_STATES_ROOT_DIRNAME = "model"

# A temporary flag to enable the new idempotent saving framework.
_SAVING_V3_ENABLED = threading.local()
_SAVING_V3_ENABLED.value = False

ATTR_SKIPLIST = frozenset(
    {
        "_self_tracked_trackables",
        "_layer_call_argspecs",
        "_self_unconditional_dependency_names",
        "_output_layers",
        "_input_layers",
        "submodules",
        "weights",
        "non_trainable_weights",
        "trainable_weights",
        "variables",
        "non_trainable_variables",
        "trainable_variables",
        "updates",  # Would raise a warning if visited.
        "state_updates",  # Would raise a warning if visited.
    }
)


def save_model(model, filepath):
    """Save a zip-archive representing a Keras model to the given filepath.

    The zip-based archive contains the following structure:

    - JSON-based configuration file (config.json): Records of model, layer, and
        other trackables' configuration.
    - NPZ-based trackable state files, found in respective directories, such as
        model/states.npz, model/dense_layer/states.npz, etc.
    - Metadata file (this is a TODO).

    The states of Keras trackables (layers, optimizers, loss, and metrics) are
    automatically saved as long as they can be discovered through the attributes
    returned by `dir(Model)`. Typically, the state includes the variables
    associated with the trackable, but some specially purposed layers may
    contain more such as the vocabularies stored in the hashmaps. The trackables
    define how their states are saved by exposing `save_state()` and
    `load_state()` APIs.

    For the case of layer states, the variables will be visited as long as
    they are either 1) referenced via layer attributes, or 2) referenced via a
    container (list, tuple, or dict), and the container is referenced via a
    layer attribute.
    """
    if not filepath.endswith(".keras"):
        raise ValueError(
            "Invalid filename: expected a `.keras` extension. "
            f"Received: filepath={filepath}"
        )
    if not model.built:
        warnings.warn(
            "You are saving a model that has not yet been built. "
            "It might not contain any weights yet. "
            "Consider building the model first by calling it "
            "on some data.",
            stacklevel=2,
        )
    saving_v3_enabled_value = getattr(_SAVING_V3_ENABLED, "value", False)
    _SAVING_V3_ENABLED.value = True

    serialized_model_dict = serialize_keras_object(model)
    config_json = json.dumps(serialized_model_dict)
    # TODO(fchollet): consider saving dependencies list / versions in metadata.
    metadata_json = json.dumps(
        {
            "keras_version": keras.__version__,
            "date_saved": datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S"),
        }
    )

    # Use a temporary directory for the storing files prior to zipping.
    temp_path = _get_temp_dir()
    try:
        # Write files locally before zipping.
        with open(tf.io.gfile.join(temp_path, _METADATA_FILENAME), "w") as f:
            f.write(metadata_json)
        with open(tf.io.gfile.join(temp_path, _CONFIG_FILENAME), "w") as f:
            f.write(config_json)
        _save_state(
            model, tf.io.gfile.join(temp_path, _STATES_ROOT_DIRNAME), set()
        )

        # Zip local files into an archive.
        with zipfile.ZipFile(filepath, "w") as zipfile_to_save:
            _write_recursively(zipfile_to_save, temp_path, "")
        _print_archive(zipfile_to_save, "saving")
    except Exception as e:
        raise e
    finally:
        _SAVING_V3_ENABLED.value = saving_v3_enabled_value
        # Remove the directory temporarily used.
        tf.io.gfile.rmtree(temp_path)


def load_model(filepath, custom_objects=None):
    """Load a zip archive representing a Keras model."""
    if not filepath.endswith(".keras"):
        raise ValueError(
            "Invalid filename: expected a `.keras` extension. "
            f"Received: filepath={filepath}"
        )
    saving_v3_enabled_value = getattr(_SAVING_V3_ENABLED, "value", False)
    _SAVING_V3_ENABLED.value = True
    temp_path = _get_temp_dir()
    try:
        with zipfile.ZipFile(filepath, "r") as zipfile_to_load:
            _print_archive(zipfile_to_load, "loading")
            zipfile_to_load.extractall(temp_path)

        with open(tf.io.gfile.join(temp_path, _CONFIG_FILENAME), "r") as f:
            config_json = f.read()
        # Note: we should NOT use a custom JSON decoder. Anything that
        # needs custom decoding must be handled in deserialize_keras_object.
        config_dict = json.loads(config_json)
        # Construct the model from the configuration file in the archive.
        model = deserialize_keras_object(config_dict, custom_objects)
        _load_state(model, tf.io.gfile.join(temp_path, _STATES_ROOT_DIRNAME))
    except Exception as e:
        raise e
    else:
        return model
    finally:
        _SAVING_V3_ENABLED.value = saving_v3_enabled_value
        if tf.io.gfile.exists(temp_path):
            tf.io.gfile.rmtree(temp_path)


def _write_recursively(zipfile_to_save, system_path, zip_path):
    if not tf.io.gfile.isdir(system_path):
        zipfile_to_save.write(system_path, zip_path)
    else:
        for file_name in tf.io.gfile.listdir(system_path):
            system_file_path = tf.io.gfile.join(system_path, file_name)
            zip_file_path = tf.io.gfile.join(zip_path, file_name)
            _write_recursively(zipfile_to_save, system_file_path, zip_file_path)


def _save_state(trackable, temp_path, saved_trackables):
    # If the trackable has already been saved, skip it.
    if id(trackable) in saved_trackables:
        return

    # TODO(rchao): Make `.get_state()` and `.save_state()` exported methods.
    if hasattr(trackable, "_save_state"):
        if not tf.io.gfile.exists(temp_path):
            tf.io.gfile.makedirs(temp_path)
        trackable._save_state(temp_path)
        saved_trackables.add(id(trackable))

    # Recursively save state of children trackables (layers, optimizers, etc.)
    for child_attr in dir(trackable):
        if child_attr in ATTR_SKIPLIST:
            continue
        try:
            child_obj = getattr(trackable, child_attr)
        except Exception:
            # Avoid raising the exception when visiting the attributes.
            continue
        if _is_keras_trackable(child_obj):
            _save_state(
                child_obj,
                tf.io.gfile.join(temp_path, child_attr),
                saved_trackables,
            )
        elif isinstance(child_obj, (list, dict, tuple)):
            _save_container_state(
                child_obj,
                tf.io.gfile.join(temp_path, child_attr),
                saved_trackables,
            )


def _load_state(trackable, temp_path):
    if hasattr(trackable, "_load_state"):
        trackable._load_state(temp_path)

    # Recursively load states for Keras trackables such as layers/optimizers.
    for child_attr in dir(trackable):
        if child_attr in ATTR_SKIPLIST:
            continue
        try:
            child_obj = getattr(trackable, child_attr)
        except Exception:
            # Avoid raising exceptions when visiting attributes.
            continue
        if _is_keras_trackable(child_obj):
            _load_state(
                child_obj,
                tf.io.gfile.join(temp_path, child_attr),
            )
        elif isinstance(child_obj, (list, dict, tuple)):
            _load_container_state(
                child_obj,
                tf.io.gfile.join(temp_path, child_attr),
            )


def _save_container_state(container, temp_path, saved_trackables):
    for trackable in container:
        if _is_keras_trackable(trackable):
            _save_state(
                trackable,
                tf.io.gfile.join(temp_path, trackable.name),
                saved_trackables,
            )


def _load_container_state(container, temp_path):
    for trackable in container:
        if _is_keras_trackable(trackable):
            _load_state(
                trackable,
                tf.io.gfile.join(temp_path, trackable.name),
            )


def _get_temp_dir():
    temp_dir = tempfile.mkdtemp()
    try:
        testfile = tempfile.TemporaryFile(dir=temp_dir)
        testfile.close()
        stats = os.statvfs(temp_dir)
        available_space = stats.f_frsize * stats.f_bavail
    except OSError:
        # Non-writable
        available_space = 0
    if available_space < 2000000000:
        # Fallback on RAM if disk is nonwritable or if less than 2GB available.
        temp_dir = f"ram://{uuid.uuid4()}"
        tf.io.gfile.mkdir(temp_dir)
    return temp_dir


def _print_archive(zipfile, action):
    # TODO(fchollet): move to debugging logs.
    io_utils.print_msg(f"Keras model {action}:")
    # Same as `ZipFile.printdir()` except for using Keras' printing utility.
    io_utils.print_msg(
        "%-46s %19s %12s" % ("File Name", "Modified    ", "Size")
    )
    for zinfo in zipfile.filelist:
        date = "%d-%02d-%02d %02d:%02d:%02d" % zinfo.date_time[:6]
        io_utils.print_msg(
            "%-46s %s %12d" % (zinfo.filename, date, zinfo.file_size)
        )


def _is_keras_trackable(obj):
    from keras.metrics import base_metric  # To avoid circular import

    return isinstance(
        obj,
        (
            base_layer.Layer,
            optimizer.Optimizer,
            base_metric.Metric,
            losses.Loss,
        ),
    )
