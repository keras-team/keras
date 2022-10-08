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
import tempfile
import threading
import warnings
import zipfile

import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras import losses
from keras.engine import base_layer
from keras.optimizers.optimizer_experimental import optimizer
from keras.saving.experimental.serialization_lib import deserialize_keras_object
from keras.saving.experimental.serialization_lib import serialize_keras_object
from keras.utils import generic_utils
from keras.utils import io_utils

try:
    import h5py
except ImportError:
    h5py = None

# isort: off

_CONFIG_FILENAME = "config.json"
_METADATA_FILENAME = "metadata.json"
_VARS_FNAME = "variables"
_ASSETS_DIRNAME = "assets"

# A temporary flag to enable the new idempotent saving framework.
_SAVING_V3_ENABLED = threading.local()
_SAVING_V3_ENABLED.value = False

ATTR_SKIPLIST = frozenset(
    {
        "__dict__",
        "_self_tracked_trackables",
        "_layer_call_argspecs",
        "_self_unconditional_dependency_names",
        "_output_layers",
        "_input_layers",
        "_trainable_weights",
        "_non_trainable_weights",
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


def save_model(model, filepath, weights_format="h5", use_zip=True):
    """Save an archive representing a Keras model to the given filepath.

    The zip-based archive contains the following structure:

    - JSON configuration file (`config.json`): Records of model, layer, and
        other object configurations.
    - Npz or h5 model variables file (`variables.npz` or `variables.h5`).
    - Assets files (if any) found in the `assets/` directory structure,
        which mirrors the model's inner structure.
    - JSON metadata file (`metdata.json`).

    The states of Keras trackables (layers, optimizers, loss, and metrics) are
    automatically saved as long as they can be discovered through the attributes
    returned by `dir(model)`. Typically, the state includes the variables
    associated with the trackable, but some specially purposed layers may
    contain more such as the vocabularies stored in the hashmaps. The trackables
    define how their asset state is saved by exposing `save_assets()` and
    `load_assets()` APIs.

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
    if weights_format == "h5" and h5py is None:
        raise ImportError(
            "h5py must be installed in order to save a model in hdf5 format."
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
    if use_zip:
        # Use a temporary directory for the storing files prior to zipping.
        write_path = _get_temp_dir()
    else:
        tf.io.gfile.makedirs(filepath)
        write_path = filepath
    try:
        # Write files locally before zipping.
        with open(tf.io.gfile.join(write_path, _METADATA_FILENAME), "w") as f:
            f.write(metadata_json)
        with open(tf.io.gfile.join(write_path, _CONFIG_FILENAME), "w") as f:
            f.write(config_json)

        weights_path = tf.io.gfile.join(write_path, _VARS_FNAME)
        assets_path = tf.io.gfile.join(write_path, _ASSETS_DIRNAME)

        if weights_format == "h5":
            weights_store = H5IOStore(weights_path, mode="w")
        elif weights_format == "npz":
            weights_store = NpzIOStore(weights_path, mode="w")
        else:
            raise ValueError(
                "Unknown `weights_format`. Expected 'h5' or 'npz'.  "
                f"Received: {weights_format}"
            )
        _save_state(
            model,
            weights_handler=weights_store,
            assets_handler=DiskIOStore(assets_path),
            inner_path="",
            visited_trackables=set(),
        )
        weights_store.close()

        if use_zip:
            # Zip local files into an archive.
            with zipfile.ZipFile(filepath, "w") as zipfile_to_save:
                _write_to_zip_recursively(zipfile_to_save, write_path, "")
    except Exception as e:
        raise e
    finally:
        _SAVING_V3_ENABLED.value = saving_v3_enabled_value
        if use_zip and tf.io.gfile.exists(write_path):
            # Remove the directory temporarily used.
            tf.io.gfile.rmtree(write_path)


def load_model(filepath, custom_objects=None):
    """Load an archive representing a Keras model."""
    if not filepath.endswith(".keras"):
        raise ValueError(
            "Invalid filename: expected a `.keras` extension. "
            f"Received: filepath={filepath}"
        )
    use_zip = not tf.io.gfile.isdir(filepath)

    saving_v3_enabled_value = getattr(_SAVING_V3_ENABLED, "value", False)
    _SAVING_V3_ENABLED.value = True

    if use_zip:
        read_path = _get_temp_dir()
    else:
        read_path = filepath
    try:
        if use_zip:
            with zipfile.ZipFile(filepath, "r") as zipfile_to_load:
                zipfile_to_load.extractall(read_path)

        with open(tf.io.gfile.join(read_path, _CONFIG_FILENAME), "r") as f:
            config_json = f.read()
        # Note: we should NOT use a custom JSON decoder. Anything that
        # needs custom decoding must be handled in deserialize_keras_object.
        config_dict = json.loads(config_json)
        # Construct the model from the configuration file in the archive.
        model = deserialize_keras_object(config_dict, custom_objects)

        weights_path = tf.io.gfile.join(read_path, _VARS_FNAME)
        if tf.io.gfile.exists(weights_path + ".h5"):
            weights_format = "h5"
            if h5py is None:
                raise ImportError(
                    "h5py must be installed in order to save "
                    "a model in hdf5 format."
                )
        elif tf.io.gfile.exists(weights_path + ".npz"):
            weights_format = "npz"

        if weights_format == "h5":
            weights_store = H5IOStore(weights_path, mode="r")
        elif weights_format == "npz":
            weights_store = NpzIOStore(weights_path, mode="r")
        else:
            raise ValueError(
                f"Expected a {weights_path}.h5 or {weights_path}.npz file."
            )

        assets_path = tf.io.gfile.join(read_path, _ASSETS_DIRNAME)
        _load_state(
            model,
            weights_handler=weights_store,
            assets_handler=DiskIOStore(assets_path),
            inner_path="",
            visited_trackables=set(),
        )
        weights_store.close()
    except Exception as e:
        raise e
    else:
        return model
    finally:
        _SAVING_V3_ENABLED.value = saving_v3_enabled_value
        if use_zip and tf.io.gfile.exists(read_path):
            tf.io.gfile.rmtree(read_path)


def _write_to_zip_recursively(zipfile_to_save, system_path, zip_path):
    if not tf.io.gfile.isdir(system_path):
        zipfile_to_save.write(system_path, zip_path)
    else:
        for file_name in tf.io.gfile.listdir(system_path):
            system_file_path = tf.io.gfile.join(system_path, file_name)
            zip_file_path = tf.io.gfile.join(zip_path, file_name)
            _write_to_zip_recursively(
                zipfile_to_save, system_file_path, zip_file_path
            )


def _save_state(
    trackable, weights_handler, assets_handler, inner_path, visited_trackables
):
    # If the trackable has already been saved, skip it.
    if id(trackable) in visited_trackables:
        return

    # TODO(fchollet): better name?
    if hasattr(trackable, "_save_own_variables"):
        trackable._save_own_variables(weights_handler.make(inner_path))
    if hasattr(trackable, "_save_assets"):
        trackable._save_assets(assets_handler.make(inner_path))

    visited_trackables.add(id(trackable))

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
                weights_handler,
                assets_handler,
                inner_path=tf.io.gfile.join(inner_path, child_attr),
                visited_trackables=visited_trackables,
            )
        elif isinstance(child_obj, (list, dict, tuple)):
            _save_container_state(
                child_obj,
                weights_handler,
                assets_handler,
                inner_path=tf.io.gfile.join(inner_path, child_attr),
                visited_trackables=visited_trackables,
            )


def _load_state(
    trackable, weights_handler, assets_handler, inner_path, visited_trackables
):
    if id(trackable) in visited_trackables:
        return

    if hasattr(trackable, "_load_own_variables"):
        trackable._load_own_variables(weights_handler.get(inner_path))
    if hasattr(trackable, "_load_assets"):
        trackable._load_assets(assets_handler.get(inner_path))

    visited_trackables.add(id(trackable))

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
                weights_handler,
                assets_handler,
                inner_path=tf.io.gfile.join(inner_path, child_attr),
                visited_trackables=visited_trackables,
            )
        elif isinstance(child_obj, (list, dict, tuple)):
            _load_container_state(
                child_obj,
                weights_handler,
                assets_handler,
                inner_path=tf.io.gfile.join(inner_path, child_attr),
                visited_trackables=visited_trackables,
            )


def _save_container_state(
    container, weights_handler, assets_handler, inner_path, visited_trackables
):
    used_names = {}
    for trackable in container:
        if _is_keras_trackable(trackable):
            # Do NOT address the trackable via `trackable.name`, since
            # names are usually autogenerated and thus not reproducible
            # (i.e. they may vary across two instances of the same model).
            name = generic_utils.to_snake_case(trackable.__class__.__name__)
            if name in used_names:
                used_names[name] += 1
                name = f"{name}_{used_names[name]}"
            else:
                used_names[name] = 0
            _save_state(
                trackable,
                weights_handler,
                assets_handler,
                inner_path=tf.io.gfile.join(inner_path, name),
                visited_trackables=visited_trackables,
            )


def _load_container_state(
    container, weights_handler, assets_handler, inner_path, visited_trackables
):
    used_names = {}
    for trackable in container:
        if _is_keras_trackable(trackable):
            name = generic_utils.to_snake_case(trackable.__class__.__name__)
            if name in used_names:
                used_names[name] += 1
                name = f"{name}_{used_names[name]}"
            else:
                used_names[name] = 0
            _load_state(
                trackable,
                weights_handler,
                assets_handler,
                inner_path=tf.io.gfile.join(inner_path, name),
                visited_trackables=visited_trackables,
            )


class DiskIOStore:
    def __init__(self, root_path, mode=None):
        self.base_directory = root_path

    def make(self, path):
        if not path:
            return self.base_directory
        path = tf.io.gfile.join(self.base_directory, path)
        if not tf.io.gfile.exists(path):
            tf.io.gfile.makedirs(path)
        return path

    def get(self, path):
        if not path:
            return self.base_directory
        path = tf.io.gfile.join(self.base_directory, path)
        if tf.io.gfile.exists(path):
            return path
        return None

    def close(self):
        pass


class H5IOStore:
    def __init__(self, root_path, mode="r"):
        self.h5_file = h5py.File(root_path + ".h5", mode=mode)

    def make(self, path):
        if not path:
            return self.h5_file.create_group("vars")
        return self.h5_file.create_group(path).create_group("vars")

    def get(self, path):
        if not path:
            return self.h5_file["vars"]
        if path in self.h5_file and "vars" in self.h5_file[path]:
            return self.h5_file[path]["vars"]
        return {}

    def close(self):
        self.h5_file.close()


class NpzIOStore:
    def __init__(self, root_path, mode="r"):
        self.root_path = root_path
        self.mode = mode
        if mode == "w":
            self.contents = {}
        else:
            f = open(root_path + ".npz", mode="rb")
            self.contents = np.load(f)
            f.close()

    def make(self, path):
        if not path:
            self.contents["vars"] = {}
            return self.contents["vars"]
        self.contents[path] = {"vars": {}}
        return self.contents[path]["vars"]

    def get(self, path):
        if not path:
            if "vars" in self.contents:
                return self.contents["vars"]
            return {}
        if path in self.contents and "vars" in self.contents[path]:
            return self.contents[path]["vars"]
        return {}

    def close(self):
        if self.mode == "w":
            f = open(self.root_path + ".npz", mode="wb")
            np.savez(f, **self.contents)
            f.close()


def _get_temp_dir():
    temp_dir = tempfile.mkdtemp()
    testfile = tempfile.TemporaryFile(dir=temp_dir)
    testfile.close()
    return temp_dir


def _print_h5_file(h5_file, prefix="", action=None):
    if not prefix:
        print(f"Keras weights file ({h5_file}) {action}:")
    if not hasattr(h5_file, "keys"):
        return
    for key in h5_file.keys():
        print(f"...{prefix}{key}")
        _print_h5_file(h5_file[key], prefix=prefix + "...")


def _print_zip_file(zipfile, action):
    # TODO(fchollet): move to debugging logs.
    io_utils.print_msg(f"Keras model archive {action}:")
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
