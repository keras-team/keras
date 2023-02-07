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
"""Public API surface for saving APIs."""

import os
import zipfile

import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export

from keras.saving import saving_lib
from keras.saving.legacy import save as legacy_sm_saving_lib
from keras.utils import io_utils

try:
    import h5py
except ImportError:
    h5py = None


@keras_export("keras.saving.save_model", "keras.models.save_model")
def save_model(model, filepath, overwrite=True, save_format=None, **kwargs):
    """Saves a model as a TensorFlow SavedModel or HDF5 file.

    See the [Serialization and Saving guide](
        https://keras.io/guides/serialization_and_saving/) for details.

    Args:
        model: Keras model instance to be saved.
        filepath: `str` or `pathlib.Path` object. Path where to save the model.
        overwrite: Whether we should overwrite any existing model at the target
            location, or instead ask the user via an interactive prompt.
        save_format: Either `"keras"`, `"tf"`, `"h5"`,
            indicating whether to save the model
            in the native Keras format (`.keras`),
            in the TensorFlow SavedModel format (referred to as "SavedModel"
            below), or in the legacy HDF5 format (`.h5`).
            Defaults to `"tf"` in TF 2.X, and `"h5"` in TF 1.X.

    SavedModel format arguments:
        include_optimizer: Only applied to SavedModel and legacy HDF5 formats.
            If False, do not save the optimizer state. Defaults to True.
        signatures: Only applies to SavedModel format. Signatures to save
            with the SavedModel. See the `signatures` argument in
            `tf.saved_model.save` for details.
        options: Only applies to SavedModel format.
            `tf.saved_model.SaveOptions` object that specifies SavedModel
            saving options.
        save_traces: Only applies to SavedModel format. When enabled, the
            SavedModel will store the function traces for each layer. This
            can be disabled, so that only the configs of each layer are stored.
            Defaults to `True`. Disabling this will decrease serialization time
            and reduce file size, but it requires that all custom layers/models
            implement a `get_config()` method.

    Example:

    ```python
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_shape=(3,)),
        tf.keras.layers.Softmax()])
    model.save("model.keras")
    loaded_model = tf.keras.saving.load_model("model.keras")
    x = tf.random.uniform((10, 3))
    assert np.allclose(model.predict(x), loaded_model.predict(x))
    ```

    Note that `model.save()` is an alias for `tf.keras.saving.save_model()`.

    The SavedModel or HDF5 file contains:

    - The model's configuration (architecture)
    - The model's weights
    - The model's optimizer's state (if any)

    Thus models can be reinstantiated in the exact same state, without any of
    the code used for model definition or training.

    Note that the model weights may have different scoped names after being
    loaded. Scoped names include the model/layer names, such as
    `"dense_1/kernel:0"`. It is recommended that you use the layer properties to
    access specific variables, e.g. `model.get_layer("dense_1").kernel`.

    __SavedModel serialization format__

    With `save_format="tf"`, the model and all trackable objects attached
    to the it (e.g. layers and variables) are saved as a TensorFlow SavedModel.
    The model config, weights, and optimizer are included in the SavedModel.
    Additionally, for every Keras layer attached to the model, the SavedModel
    stores:

    * The config and metadata -- e.g. name, dtype, trainable status
    * Traced call and loss functions, which are stored as TensorFlow
      subgraphs.

    The traced functions allow the SavedModel format to save and load custom
    layers without the original class definition.

    You can choose to not save the traced functions by disabling the
    `save_traces` option. This will decrease the time it takes to save the model
    and the amount of disk space occupied by the output SavedModel. If you
    enable this option, then you _must_ provide all custom class definitions
    when loading the model. See the `custom_objects` argument in
    `tf.keras.saving.load_model`.
    """
    save_format = get_save_format(filepath, save_format)
    if save_format not in ("keras", "tf", "h5", "keras_v3"):
        raise ValueError(
            "Unknown `save_format` argument. Expected one of "
            "'keras', 'tf', or 'h5'. "
            f"Received: save_format{save_format}"
        )
    if save_format == "keras_v3" or (
        saving_lib.saving_v3_enabled() and save_format == "keras"
    ):
        # If file exists and should not be overwritten.
        try:
            exists = os.path.exists(filepath)
        except TypeError:
            exists = False
        if exists and not overwrite:
            proceed = io_utils.ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return
        if kwargs:
            raise ValueError(
                "The following argument(s) are not supported "
                f"with the native Keras format: {list(kwargs.keys())}"
            )
        saving_lib.save_model(model, filepath)
    else:
        # Legacy case
        return legacy_sm_saving_lib.save_model(
            model,
            filepath,
            overwrite=overwrite,
            save_format=save_format,
            **kwargs,
        )


@keras_export("keras.saving.load_model", "keras.models.load_model")
def load_model(
    filepath, custom_objects=None, compile=True, safe_mode=True, **kwargs
):
    """Loads a model saved via `model.save()`.

    Args:
        filepath: `str` or `pathlib.Path` object, path to the saved model file.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model after loading.
        safe_mode: Boolean, whether to disallow unsafe `lambda` deserialization.
            When `safe_mode=False`, loading an object has the potential to
            trigger arbitrary code execution. This argument is only
            applicable to the Keras v3 model format. Defaults to True.

    SavedModel format arguments:
        options: Only applies to SavedModel format.
            Optional `tf.saved_model.LoadOptions` object that specifies
            SavedModel loading options.

    Returns:
        A Keras model instance. If the original model was compiled,
        and the argument `compile=True` is set, then the returned model
        will be compiled. Otherwise, the model will be left uncompiled.

    Example:

    ```python
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_shape=(3,)),
        tf.keras.layers.Softmax()])
    model.save("model.keras")
    loaded_model = tf.keras.saving.load_model("model.keras")
    x = tf.random.uniform((10, 3))
    assert np.allclose(model.predict(x), loaded_model.predict(x))
    ```

    Note that the model variables may have different name values
    (`var.name` property, e.g. `"dense_1/kernel:0"`) after being reloaded.
    It is recommended that you use layer attributes to
    access specific variables, e.g. `model.get_layer("dense_1").kernel`.
    """
    if str(filepath).endswith(".keras") and zipfile.is_zipfile(filepath):
        if kwargs:
            raise ValueError(
                "The following argument(s) are not supported "
                f"with the native Keras format: {list(kwargs.keys())}"
            )
        return saving_lib.load_model(
            filepath,
            custom_objects=custom_objects,
            compile=compile,
            safe_mode=safe_mode,
        )

    # Legacy case.
    return legacy_sm_saving_lib.load_model(
        filepath, custom_objects=custom_objects, compile=compile, **kwargs
    )


def save_weights(model, filepath, overwrite=True, **kwargs):
    if str(filepath).endswith(".weights.h5"):
        # If file exists and should not be overwritten.
        try:
            exists = os.path.exists(filepath)
        except TypeError:
            exists = False
        if exists and not overwrite:
            proceed = io_utils.ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return
        saving_lib.save_weights_only(model, filepath)
    else:
        legacy_sm_saving_lib.save_weights(
            model, filepath, overwrite=overwrite, **kwargs
        )


def load_weights(model, filepath, skip_mismatch=False, **kwargs):
    if str(filepath).endswith(".keras") and zipfile.is_zipfile(filepath):
        saving_lib.load_weights_only(
            model, filepath, skip_mismatch=skip_mismatch
        )
    elif str(filepath).endswith(".weights.h5"):
        saving_lib.load_weights_only(
            model, filepath, skip_mismatch=skip_mismatch
        )
    else:
        return legacy_sm_saving_lib.load_weights(
            model, filepath, skip_mismatch=skip_mismatch, **kwargs
        )


def get_save_format(filepath, save_format):
    if saving_lib.saving_v3_enabled():
        default_format = "keras"
    elif tf.__internal__.tf2.enabled():
        default_format = "tf"
    else:
        default_format = "h5"

    if (h5py is not None and isinstance(filepath, h5py.File)) or str(
        filepath
    ).endswith((".h5", ".hdf5")):
        if save_format and save_format != "h5":
            raise ValueError(
                "Provided `save_format` is inconsistent with `filepath`. "
                f"Received: save_format='{save_format}', filepath='{filepath}'"
            )
        save_format = "h5"

    return save_format or default_format
