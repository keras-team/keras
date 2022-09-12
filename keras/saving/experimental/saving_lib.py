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

import json
import os
import tempfile
import uuid
import zipfile

import tensorflow.compat.v2 as tf
from absl import logging

from keras import losses
from keras.engine import base_layer
from keras.optimizers.optimizer_experimental import optimizer
from keras.saving.experimental.serialization_lib import deserialize_keras_object
from keras.saving.experimental.serialization_lib import serialize_keras_object
from keras.utils import io_utils

# isort: off

_SELF_DIRNAME = "self"
_CONFIG_FILENAME = "config.json"
_STATES_ROOT_DIRNAME = "model"

# A temporary flag to enable the new idempotent saving framework.
_ENABLED = False


def _print_archive(zipfile, action):
    # TODO(fchollet): move to debugging logs.
    io_utils.print_msg(f"Keras model is being {action} an archive:")
    # Same as `ZipFile.printdir()` except for using Keras' printing utility.
    io_utils.print_msg(
        "%-46s %19s %12s" % ("File Name", "Modified    ", "Size")
    )
    for zinfo in zipfile.filelist:
        date = "%d-%02d-%02d %02d:%02d:%02d" % zinfo.date_time[:6]
        io_utils.print_msg(
            "%-46s %s %12d" % (zinfo.filename, date, zinfo.file_size)
        )


def _is_keras_trackable(object):
    from keras.metrics import base_metric  # To avoid circular import

    return (
        isinstance(object, base_layer.Layer)
        or isinstance(object, optimizer.Optimizer)
        or isinstance(object, base_metric.Metric)
        or isinstance(object, losses.Loss)
    )


def is_container(object):
    return (
        isinstance(object, list)
        or isinstance(object, tuple)
        or isinstance(object, dict)
    )


def _extract_dir(zipfile_to_load, root_system_path, zip_dir):
    for zip_path in zipfile_to_load.namelist():
        if zip_path.startswith(zip_dir):
            created_path = zipfile_to_load.extract(zip_path, root_system_path)
            logging.debug(
                f"Extracting {zip_path} into {root_system_path}. "
                f"Created {created_path}."
            )


def _load_state(trackable, zip_dirpath, temp_path, zipfile_to_load):
    states_dirpath = tf.io.gfile.join(zip_dirpath, _SELF_DIRNAME)
    # Extract the whole directory that represents the states of the trackable
    # into a temporary directory to be removed at the end.
    _extract_dir(zipfile_to_load, temp_path, states_dirpath)
    dirpath_to_load_state = tf.io.gfile.join(temp_path, states_dirpath)
    # TODO(rchao): Make `.set_state()` and `.load_state()` exported methods
    # and remove the attr check.
    if hasattr(trackable, "_load_state"):
        trackable._load_state(dirpath_to_load_state)
    if tf.io.gfile.exists(dirpath_to_load_state):
        tf.io.gfile.rmtree(dirpath_to_load_state)

    # Recursively load states for Keras trackables such as layers/optimizers.
    for child_attr in dir(trackable):
        if (
            child_attr == "_self_tracked_trackables"
            or child_attr == "_layer_call_argspecs"
            or child_attr == "_output_layers"
        ):
            # Avoid certain attribute names to allow readable state file paths,
            # e.g., `layers`.
            continue
        try:
            child_obj = getattr(trackable, child_attr)
        except Exception:
            # Avoid raising exceptions when visiting attributes.
            continue
        if _is_keras_trackable(child_obj):
            _load_state(
                child_obj,
                tf.io.gfile.join(zip_dirpath, child_attr),
                temp_path,
                zipfile_to_load,
            )
        elif is_container(child_obj):
            _load_container_state(
                child_obj,
                tf.io.gfile.join(zip_dirpath, child_attr),
                temp_path,
                zipfile_to_load,
            )


def _load_container_state(container, zip_dirpath, temp_path, zipfile_to_load):
    for trackable in container:
        if _is_keras_trackable(trackable):
            _load_state(
                trackable,
                tf.io.gfile.join(zip_dirpath, trackable.name),
                temp_path,
                zipfile_to_load,
            )


def load_model(filepath, custom_objects=None):
    """Load a zip archive representing a Keras model."""
    if not filepath.endswith(".keras"):
        raise ValueError(
            "Invalid filename: expected a `.keras` extension. "
            f"Received: filepath={filepath}"
        )
    temp_path = _get_temp_dir()
    try:
        with zipfile.ZipFile(filepath, "r") as zipfile_to_load:
            _print_archive(zipfile_to_load, "loaded from")
            with zipfile_to_load.open(_CONFIG_FILENAME, "r") as c:
                config_json = c.read()
            logging.debug(f"Read config: {config_json} from {c}")
            # Note: we should NOT use a custom JSON decoder. Anything that
            # needs custom decoding must be handled in deserialize_keras_object.
            config_dict = json.loads(config_json)
            # Construct the model from the configuration file in the archive.
            model = deserialize_keras_object(config_dict, custom_objects)
            _load_state(model, _STATES_ROOT_DIRNAME, temp_path, zipfile_to_load)
    except Exception as e:
        raise e
    else:
        return model
    finally:
        if tf.io.gfile.exists(temp_path):
            tf.io.gfile.rmtree(temp_path)


def _write_recursively(zipfile_to_save, system_path, zip_path):
    if not tf.io.gfile.isdir(system_path):
        zipfile_to_save.write(system_path, zip_path)
        logging.debug(f"Written {system_path} into {zip_path} in the zip.")
    else:
        for file_name in tf.io.gfile.listdir(system_path):
            system_file_path = tf.io.gfile.join(system_path, file_name)
            zip_file_path = tf.io.gfile.join(zip_path, file_name)
            _write_recursively(zipfile_to_save, system_file_path, zip_file_path)


def _save_state(
    trackable, zip_dirpath, temp_path, zipfile_to_save, saved_trackables
):
    # Check whether this trackable has been saved; if so, do not duplicate the
    # saving.
    if trackable in saved_trackables:
        return

    # TODO(rchao): Make `.get_state()` and `.save_state()` exported methods
    # and remove the attr check.
    if hasattr(trackable, "_save_state"):
        # Designate a `self` directory for the trackable object to save.
        states_dirpath = tf.io.gfile.join(temp_path, _SELF_DIRNAME)
        if not tf.io.gfile.exists(states_dirpath):
            tf.io.gfile.mkdir(states_dirpath)
        trackable._save_state(states_dirpath)
        if states_dirpath is not None:
            # Recursively write the states (represented by files inside the
            # directory) into the zip file.
            _write_recursively(
                zipfile_to_save,
                states_dirpath,
                tf.io.gfile.join(zip_dirpath, _SELF_DIRNAME),
            )
            tf.io.gfile.rmtree(states_dirpath)
        saved_trackables.add(trackable)

    # Recursively ask contained trackable (layers, optimizers,
    # etc.) to save states.
    for child_attr in dir(trackable):
        if (
            child_attr == "_self_tracked_trackables"
            or child_attr == "_layer_call_argspecs"
            or child_attr == "_output_layers"
        ):
            # Avoid certain attribute names to allow readable state file paths,
            # e.g., `layers`.
            continue
        try:
            child_obj = getattr(trackable, child_attr)
        except Exception:
            # Avoid raising the exception when visiting the attributes.
            continue
        if _is_keras_trackable(child_obj):
            _save_state(
                child_obj,
                tf.io.gfile.join(zip_dirpath, child_attr),
                temp_path,
                zipfile_to_save,
                saved_trackables,
            )
        elif is_container(child_obj):
            _save_container_state(
                child_obj,
                tf.io.gfile.join(zip_dirpath, child_attr),
                temp_path,
                zipfile_to_save,
                saved_trackables,
            )


def _save_container_state(
    container, zip_dirpath, temp_path, zipfile_to_save, saved_trackables
):
    for trackable in container:
        if _is_keras_trackable(trackable):
            _save_state(
                trackable,
                tf.io.gfile.join(zip_dirpath, trackable.name),
                temp_path,
                zipfile_to_save,
                saved_trackables,
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

    # TODO(rchao): Save the model's metadata (e.g. Keras version) in a separate
    # file in the archive.
    serialized_model_dict = serialize_keras_object(model)
    config_json = json.dumps(serialized_model_dict).encode()

    # Utilize a temporary directory for the storing files prior to zipping.
    temp_path = _get_temp_dir()

    try:
        # Save the configuration json and state files.
        with zipfile.ZipFile(filepath, "x") as zipfile_to_save:
            with zipfile_to_save.open(_CONFIG_FILENAME, "w") as c:
                c.write(config_json)
                logging.debug(f"Written config: {config_json} into {c}.")
            _save_state(
                model, _STATES_ROOT_DIRNAME, temp_path, zipfile_to_save, set()
            )
            _print_archive(zipfile_to_save, "saved in")
    except Exception as e:
        raise e
    finally:
        # Remove the directory temporarily used.
        tf.io.gfile.rmtree(temp_path)


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
