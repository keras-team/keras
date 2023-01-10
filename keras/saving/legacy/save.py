# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Keras model saving code."""

import os

import tensorflow.compat.v2 as tf

from keras import backend
from keras.saving import object_registration
from keras.saving.legacy import hdf5_format
from keras.saving.legacy import saving_utils
from keras.saving.legacy import serialization
from keras.saving.legacy.saved_model import load as saved_model_load
from keras.saving.legacy.saved_model import load_context
from keras.saving.legacy.saved_model import save as saved_model_save
from keras.saving.legacy.saved_model.utils import keras_option_scope
from keras.utils import io_utils
from keras.utils import traceback_utils

try:
    import h5py
except ImportError:
    h5py = None


@traceback_utils.filter_traceback
def save_model(
    model,
    filepath,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
    save_traces=True,
):
    """Saves a model as a TensorFlow SavedModel or HDF5 file.

    See the [Serialization and Saving
    guide](https://keras.io/guides/serialization_and_saving/) for details.

    Usage:

    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Dense(5, input_shape=(3,)),
    ...     tf.keras.layers.Softmax()])
    >>> model.save('/tmp/model')
    >>> loaded_model = tf.keras.models.load_model('/tmp/model')
    >>> x = tf.random.uniform((10, 3))
    >>> assert np.allclose(model.predict(x), loaded_model.predict(x))

    Note that `model.save()` is an alias for `tf.keras.models.save_model()`.

    The SavedModel and HDF5 file contains:

    - the model's configuration (topology)
    - the model's weights
    - the model's optimizer's state (if any)

    Thus models can be reinstantiated in the exact same state, without any of
    the code used for model definition or training.

    Note that the model weights may have different scoped names after being
    loaded. Scoped names include the model/layer names, such as
    `"dense_1/kernel:0"`. It is recommended that you use the layer properties to
    access specific variables, e.g. `model.get_layer("dense_1").kernel`.

    __SavedModel serialization format__

    Keras SavedModel uses `tf.saved_model.save` to save the model and all
    trackable objects attached to the model (e.g. layers and variables). The
    model config, weights, and optimizer are saved in the SavedModel.
    Additionally, for every Keras layer attached to the model, the SavedModel
    stores:

      * the config and metadata -- e.g. name, dtype, trainable status
      * traced call and loss functions, which are stored as TensorFlow
        subgraphs.

    The traced functions allow the SavedModel format to save and load custom
    layers without the original class definition.

    You can choose to not save the traced functions by disabling the
    `save_traces` option. This will decrease the time it takes to save the model
    and the amount of disk space occupied by the output SavedModel. If you
    enable this option, then you _must_ provide all custom class definitions
    when loading the model. See the `custom_objects` argument in
    `tf.keras.models.load_model`.

    Args:
        model: Keras model instance to be saved.
        filepath: One of the following:
          - String or `pathlib.Path` object, path where to save the model
          - `h5py.File` object where to save the model
        overwrite: Whether we should overwrite any existing model at the target
          location, or instead ask the user with a manual prompt.
        include_optimizer: If True, save optimizer's state together.
        save_format: Either 'tf' or 'h5', indicating whether to save the model
          to Tensorflow SavedModel or HDF5. Defaults to 'tf' in TF 2.X, and 'h5'
          in TF 1.X.
        signatures: Signatures to save with the SavedModel. Applicable to the
          'tf' format only. Please see the `signatures` argument in
          `tf.saved_model.save` for details.
        options: (only applies to SavedModel format)
          `tf.saved_model.SaveOptions` object that specifies options for saving
          to SavedModel.
        save_traces: (only applies to SavedModel format) When enabled, the
          SavedModel will store the function traces for each layer. This
          can be disabled, so that only the configs of each layer are stored.
          Defaults to `True`. Disabling this will decrease serialization time
          and reduce file size, but it requires that all custom layers/models
          implement a `get_config()` method.

    Raises:
        ImportError: If save format is hdf5, and h5py is not available.
    """

    from keras.engine import sequential

    default_format = "tf" if tf.__internal__.tf2.enabled() else "h5"
    save_format = save_format or default_format

    filepath = io_utils.path_to_string(filepath)

    # If the user has not already called fit or built the underlying metrics, we
    # should do that before saving to ensure the metric names have all
    # appropriate name transformations applied.
    saving_utils.try_build_compiled_arguments(model)

    if (
        save_format == "h5"
        or (h5py is not None and isinstance(filepath, h5py.File))
        or saving_utils.is_hdf5_filepath(filepath)
    ):
        # TODO(b/130258301): add utility method for detecting model type.
        if not model._is_graph_network and not isinstance(
            model, sequential.Sequential
        ):
            raise NotImplementedError(
                "Saving the model to HDF5 format requires the model to be a "
                "Functional model or a Sequential model. It does not work for "
                "subclassed models, because such models are defined via the "
                "body of a Python method, which isn't safely serializable. "
                "Consider saving to the Tensorflow SavedModel format (by "
                'setting save_format="tf") or using `save_weights`.'
            )
        hdf5_format.save_model_to_hdf5(
            model, filepath, overwrite, include_optimizer
        )
    else:
        with serialization.SharedObjectSavingScope():
            with keras_option_scope(
                save_traces=save_traces, in_tf_saved_model_scope=True
            ):
                saved_model_save.save(
                    model,
                    filepath,
                    overwrite,
                    include_optimizer,
                    signatures,
                    options,
                    save_traces,
                )


@traceback_utils.filter_traceback
def load_model(filepath, custom_objects=None, compile=True, options=None):
    """Loads a model saved via `model.save()`.

    Usage:

    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Dense(5, input_shape=(3,)),
    ...     tf.keras.layers.Softmax()])
    >>> model.save('/tmp/model')
    >>> loaded_model = tf.keras.models.load_model('/tmp/model')
    >>> x = tf.random.uniform((10, 3))
    >>> assert np.allclose(model.predict(x), loaded_model.predict(x))

    Note that the model weights may have different scoped names after being
    loaded. Scoped names include the model/layer names, such as
    `"dense_1/kernel:0"`. It is recommended that you use the layer properties to
    access specific variables, e.g. `model.get_layer("dense_1").kernel`.

    Args:
        filepath: One of the following:
            - String or `pathlib.Path` object, path to the saved model
            - `h5py.File` object from which to load the model
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model
            after loading.
        options: Optional `tf.saved_model.LoadOptions` object that specifies
          options for loading from SavedModel.

    Returns:
        A Keras model instance. If the original model was compiled, and saved
        with the optimizer, then the returned model will be compiled. Otherwise,
        the model will be left uncompiled. In the case that an uncompiled model
        is returned, a warning is displayed if the `compile` argument is set to
        `True`.

    Raises:
        ImportError: if loading from an hdf5 file and h5py is not available.
        IOError: In case of an invalid savefile.
    """
    with serialization.SharedObjectLoadingScope():
        with object_registration.CustomObjectScope(custom_objects or {}):
            with keras_option_scope(
                save_traces=False, in_tf_saved_model_scope=True
            ):
                with load_context.load_context(options):
                    filepath_str = io_utils.path_to_string(filepath)
                    if isinstance(filepath_str, str):
                        if not tf.io.gfile.exists(filepath_str):
                            raise IOError(
                                f"No file or directory found at {filepath_str}"
                            )

                        if tf.io.gfile.isdir(filepath_str):
                            return saved_model_load.load(
                                filepath_str, compile, options
                            )
                        else:
                            if h5py is None:
                                raise ImportError(
                                    "Filepath looks like a hdf5 file but h5py"
                                    "is not available."
                                    f" filepath={filepath_str}"
                                )
                            return hdf5_format.load_model_from_hdf5(
                                tf.io.gfile.GFile(filepath_str, mode="rb"),
                                custom_objects,
                                compile,
                            )
                    elif h5py is not None and isinstance(filepath, h5py.File):
                        return hdf5_format.load_model_from_hdf5(
                            filepath, custom_objects, compile
                        )

    raise IOError(
        "Unable to load model. Filepath is not an hdf5 file (or h5py is not "
        f"available) or SavedModel. Received: filepath={filepath}"
    )


def save_weights(
    model, filepath, overwrite=True, save_format=None, options=None
):
    """Saves all layer weights.

    Either saves in HDF5 or in TensorFlow format based on the `save_format`
    argument.

    When saving in HDF5 format, the weight file has:
        - `layer_names` (attribute), a list of strings
            (ordered names of model layers).
        - For every layer, a `group` named `layer.name`
            - For every such layer group, a group attribute `weight_names`,
                a list of strings
                (ordered names of weights tensor of the layer).
            - For every weight in the layer, a dataset
                storing the weight value, named after the weight tensor.

    When saving in TensorFlow format, all objects referenced by the network
    are saved in the same format as `tf.train.Checkpoint`, including any
    `Layer` instances or `Optimizer` instances assigned to object
    attributes. For networks constructed from inputs and outputs using
    `tf.keras.Model(inputs, outputs)`, `Layer` instances used by the network
    are tracked/saved automatically. For user-defined classes which inherit
    from `tf.keras.Model`, `Layer` instances must be assigned to object
    attributes, typically in the constructor. See the documentation of
    `tf.train.Checkpoint` and `tf.keras.Model` for details.

    While the formats are the same, do not mix `save_weights` and
    `tf.train.Checkpoint`. Checkpoints saved by `Model.save_weights` should
    be loaded using `Model.load_weights`. Checkpoints saved using
    `tf.train.Checkpoint.save` should be restored using the corresponding
    `tf.train.Checkpoint.restore`. Prefer `tf.train.Checkpoint` over
    `save_weights` for training checkpoints.

    The TensorFlow format matches objects and variables by starting at a
    root object, `self` for `save_weights`, and greedily matching attribute
    names. For `Model.save` this is the `Model`, and for `Checkpoint.save`
    this is the `Checkpoint` even if the `Checkpoint` has a model attached.
    This means saving a `tf.keras.Model` using `save_weights` and loading
    into a `tf.train.Checkpoint` with a `Model` attached (or vice versa)
    will not match the `Model`'s variables. See the
    [guide to training checkpoints](
    https://www.tensorflow.org/guide/checkpoint) for details on
    the TensorFlow format.

    Args:
        filepath: String or PathLike, path to the file to save the weights
            to. When saving in TensorFlow format, this is the prefix used
            for checkpoint files (multiple files are generated). Note that
            the '.h5' suffix causes weights to be saved in HDF5 format.
        overwrite: Whether to silently overwrite any existing file at the
            target location, or provide the user with a manual prompt.
        save_format: Either 'tf' or 'h5'. A `filepath` ending in '.h5' or
            '.keras' will default to HDF5 if `save_format` is `None`.
            Otherwise `None` defaults to 'tf'.
        options: Optional `tf.train.CheckpointOptions` object that specifies
            options for saving weights.

    Raises:
        ImportError: If `h5py` is not available when attempting to save in
            HDF5 format.
    """
    model._assert_weights_created()
    filepath = io_utils.path_to_string(filepath)
    filepath_is_h5 = saving_utils.is_hdf5_filepath(filepath)
    if save_format is None:
        if filepath_is_h5:
            save_format = "h5"
        else:
            save_format = "tf"
    else:
        user_format = save_format.lower().strip()
        if user_format in ("tensorflow", "tf"):
            save_format = "tf"
        elif user_format in ("hdf5", "h5", "keras"):
            save_format = "h5"
        else:
            raise ValueError(
                f"Unknown format. Received: `save_format`={save_format}. "
                'Was expecting one of {"tf", "h5"}.'
            )
    if save_format == "tf" and filepath_is_h5:
        raise ValueError(
            'save_weights got save_format="tf"/"tensorflow", but the '
            f"filepath ({filepath}) looks like an HDF5 file. "
            'Omit the ".h5"/".keras" when saving in TensorFlow format.'
        )

    if save_format == "h5" and h5py is None:
        raise ImportError(
            "`save_weights` requires h5py when saving in hdf5, but h5py is "
            "not available. Try installing h5py package."
        )
    if save_format == "tf":
        check_filepath = filepath + ".index"
    else:
        check_filepath = filepath
    # If file exists and should not be overwritten:
    if not overwrite and os.path.isfile(check_filepath):
        proceed = io_utils.ask_to_proceed_with_overwrite(check_filepath)
        if not proceed:
            return
    if save_format == "h5":
        with h5py.File(filepath, "w") as f:
            hdf5_format.save_weights_to_hdf5_group(f, model)
    else:
        if not tf.executing_eagerly():
            # Call `get_session` to initialize any uninitialized variables.
            backend.get_session()
        model._checkpoint.write(filepath, options=options)

        # Record this checkpoint so it's visible from
        # tf.train.latest_checkpoint.
        tf.__internal__.train.update_checkpoint_state(
            save_dir=os.path.dirname(filepath),
            model_checkpoint_path=filepath,
            save_relative_paths=True,
            all_model_checkpoint_paths=[filepath],
        )


def load_weights(
    model, filepath, by_name=False, skip_mismatch=False, options=None
):
    """Loads all layer weights, either from a SavedModel or H5 weights file.

    If `by_name` is False weights are loaded based on the network's
    topology. This means the architecture should be the same as when the
    weights were saved.  Note that layers that don't have weights are not
    taken into account in the topological ordering, so adding or removing
    layers is fine as long as they don't have weights.

    If `by_name` is True, weights are loaded into layers only if they share
    the same name. This is useful for fine-tuning or transfer-learning
    models where some of the layers have changed.

    Only topological loading (`by_name=False`) is supported when loading
    weights from the TensorFlow format. Note that topological loading
    differs slightly between TensorFlow and HDF5 formats for user-defined
    classes inheriting from `tf.keras.Model`: HDF5 loads based on a
    flattened list of weights, while the TensorFlow format loads based on
    the object-local names of attributes to which layers are assigned in the
    `Model`'s constructor.

    Args:
        filepath: String, path to the weights file to load. For weight files
            in TensorFlow format, this is the file prefix (the same as was
            passed to `save_weights`). This can also be a path to a
            SavedModel saved from `model.save`.
        by_name: Boolean, whether to load weights by name or by topological
            order. Only topological loading is supported for weight files in
            TensorFlow format.
        skip_mismatch: Boolean, whether to skip loading of layers where
            there is a mismatch in the number of weights, or a mismatch in
            the shape of the weight (only valid when `by_name=True`).
        options: Optional `tf.train.CheckpointOptions` object that specifies
            options for loading weights.

    Returns:
        When loading a weight file in TensorFlow format, returns the same
        status object as `tf.train.Checkpoint.restore`. When graph building,
        restore ops are run automatically as soon as the network is built
        (on first call for user-defined classes inheriting from `Model`,
        immediately if it is already built).

        When loading weights in HDF5 format, returns `None`.

    Raises:
        ImportError: If `h5py` is not available and the weight file is in
            HDF5 format.
        ValueError: If `skip_mismatch` is set to `True` when `by_name` is
            `False`.
    """
    if backend.is_tpu_strategy(model._distribution_strategy):
        if model._distribution_strategy.extended.steps_per_run > 1 and (
            not saving_utils.is_hdf5_filepath(filepath)
        ):
            spr = model._distribution_strategy.extended.steps_per_run
            raise ValueError(
                "Load weights is not implemented with TPUStrategy "
                "with `steps_per_run` greater than 1. The "
                f"`steps_per_run` is {spr}"
            )
    if skip_mismatch and not by_name:
        raise ValueError(
            "When calling model.load_weights, skip_mismatch can only be "
            "set to True when by_name is True."
        )

    filepath, save_format = _detect_save_format(filepath)
    if save_format == "tf":
        status = model._checkpoint.read(filepath, options)
        if by_name:
            raise NotImplementedError(
                "Weights may only be loaded based on topology into Models "
                "when loading TensorFlow-formatted weights "
                "(got by_name=True to load_weights)."
            )
        if not tf.executing_eagerly():
            session = backend.get_session()
            # Restore existing variables (if any) immediately, and set up a
            # streaming restore for any variables created in the future.
            tf.__internal__.tracking.streaming_restore(
                status=status, session=session
            )
        status.assert_nontrivial_match()
    else:
        status = None
        if h5py is None:
            raise ImportError(
                "`load_weights` requires h5py package when loading weights "
                "from HDF5. Try installing h5py."
            )
        if not model._is_graph_network and not model.built:
            raise ValueError(
                "Unable to load weights saved in HDF5 format into a "
                "subclassed Model which has not created its variables yet. "
                "Call the Model first, then load the weights."
            )
        model._assert_weights_created()
        with h5py.File(filepath, "r") as f:
            if "layer_names" not in f.attrs and "model_weights" in f:
                f = f["model_weights"]
            if by_name:
                hdf5_format.load_weights_from_hdf5_group_by_name(
                    f, model, skip_mismatch
                )
            else:
                hdf5_format.load_weights_from_hdf5_group(f, model)

    # Perform any layer defined finalization of the layer state.
    for layer in model.layers:
        layer.finalize_state()
    return status


def _detect_save_format(filepath):
    """Returns path to weights file and save format."""

    filepath = io_utils.path_to_string(filepath)
    if saving_utils.is_hdf5_filepath(filepath):
        return filepath, "h5"

    # Filepath could be a TensorFlow checkpoint file prefix or SavedModel
    # directory. It's possible for filepath to be both a prefix and directory.
    # Prioritize checkpoint over SavedModel.
    if _is_readable_tf_checkpoint(filepath):
        save_format = "tf"
    elif tf.saved_model.contains_saved_model(filepath):
        ckpt_path = os.path.join(
            filepath,
            tf.saved_model.VARIABLES_DIRECTORY,
            tf.saved_model.VARIABLES_FILENAME,
        )
        if _is_readable_tf_checkpoint(ckpt_path):
            filepath = ckpt_path
            save_format = "tf"
        else:
            raise ValueError(
                "Unable to load weights. filepath {} appears to be a "
                "SavedModel directory, but checkpoint either doesn't "
                "exist, or is incorrectly formatted.".format(filepath)
            )
    else:
        # Not a TensorFlow checkpoint. This filepath is likely an H5 file that
        # doesn't have the hdf5/keras extensions.
        save_format = "h5"
    return filepath, save_format


def _is_readable_tf_checkpoint(filepath):
    try:
        tf.compat.v1.train.NewCheckpointReader(filepath)
        return True
    except tf.errors.DataLossError:
        # The checkpoint is not readable in TensorFlow format.
        return False


# Inject the load_model function to keras_deps to remove the dependency
# from TFLite to Keras.
tf.__internal__.register_load_model_function(load_model)
