import os
import zipfile

from absl import logging

from keras.src.api_export import keras_export
from keras.src.legacy.saving import legacy_h5_format
from keras.src.saving import saving_lib
from keras.src.utils import file_utils
from keras.src.utils import io_utils

try:
    import h5py
except ImportError:
    h5py = None


@keras_export(["keras.saving.save_model", "keras.models.save_model"])
def save_model(model, filepath, overwrite=True, zipped=None, **kwargs):
    """Saves a model as a `.keras` file.

    Args:
        model: Keras model instance to be saved.
        filepath: `str` or `pathlib.Path` object. Path where to save the model.
        overwrite: Whether we should overwrite any existing model at the target
            location, or instead ask the user via an interactive prompt.
        zipped: Whether to save the model as a zipped `.keras`
            archive (default when saving locally), or as an unzipped directory
            (default when saving on the Hugging Face Hub).

    Example:

    ```python
    model = keras.Sequential(
        [
            keras.layers.Dense(5, input_shape=(3,)),
            keras.layers.Softmax(),
        ],
    )
    model.save("model.keras")
    loaded_model = keras.saving.load_model("model.keras")
    x = keras.random.uniform((10, 3))
    assert np.allclose(model.predict(x), loaded_model.predict(x))
    ```

    Note that `model.save()` is an alias for `keras.saving.save_model()`.

    The saved `.keras` file is a `zip` archive that contains:

    - The model's configuration (architecture)
    - The model's weights
    - The model's optimizer's state (if any)

    Thus models can be reinstantiated in the exact same state.
    """
    include_optimizer = kwargs.pop("include_optimizer", True)
    save_format = kwargs.pop("save_format", False)
    if save_format:
        if str(filepath).endswith((".h5", ".hdf5")) or str(filepath).endswith(
            ".keras"
        ):
            logging.warning(
                "The `save_format` argument is deprecated in Keras 3. "
                "We recommend removing this argument as it can be inferred "
                "from the file path. "
                f"Received: save_format={save_format}"
            )
        else:
            raise ValueError(
                "The `save_format` argument is deprecated in Keras 3. "
                "Please remove this argument and pass a file path with "
                "either `.keras` or `.h5` extension."
                f"Received: save_format={save_format}"
            )
    if kwargs:
        raise ValueError(
            "The following argument(s) are not supported: "
            f"{list(kwargs.keys())}"
        )

    # Deprecation warnings
    if str(filepath).endswith((".h5", ".hdf5")):
        logging.warning(
            "You are saving your model as an HDF5 file via "
            "`model.save()` or `keras.saving.save_model(model)`. "
            "This file format is considered legacy. "
            "We recommend using instead the native Keras format, "
            "e.g. `model.save('my_model.keras')` or "
            "`keras.saving.save_model(model, 'my_model.keras')`. "
        )

    is_hf = str(filepath).startswith("hf://")
    if zipped is None:
        zipped = not is_hf  # default behavior depends on destination

    # If file exists and should not be overwritten.
    try:
        exists = (not is_hf) and os.path.exists(filepath)
    except TypeError:
        exists = False
    if exists and not overwrite:
        proceed = io_utils.ask_to_proceed_with_overwrite(filepath)
        if not proceed:
            return

    if zipped and str(filepath).endswith(".keras"):
        return saving_lib.save_model(model, filepath)
    if not zipped:
        return saving_lib.save_model(model, filepath, zipped=False)
    if str(filepath).endswith((".h5", ".hdf5")):
        return legacy_h5_format.save_model_to_hdf5(
            model, filepath, overwrite, include_optimizer
        )
    raise ValueError(
        "Invalid filepath extension for saving. "
        "Please add either a `.keras` extension for the native Keras "
        f"format (recommended) or a `.h5` extension. "
        "Use `model.export(filepath)` if you want to export a SavedModel "
        "for use with TFLite/TFServing/etc. "
        f"Received: filepath={filepath}."
    )


@keras_export(["keras.saving.load_model", "keras.models.load_model"])
def load_model(filepath, custom_objects=None, compile=True, safe_mode=True):
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
            applicable to the Keras v3 model format. Defaults to `True`.

    Returns:
        A Keras model instance. If the original model was compiled,
        and the argument `compile=True` is set, then the returned model
        will be compiled. Otherwise, the model will be left uncompiled.

    Example:

    ```python
    model = keras.Sequential([
        keras.layers.Dense(5, input_shape=(3,)),
        keras.layers.Softmax()])
    model.save("model.keras")
    loaded_model = keras.saving.load_model("model.keras")
    x = np.random.random((10, 3))
    assert np.allclose(model.predict(x), loaded_model.predict(x))
    ```

    Note that the model variables may have different name values
    (`var.name` property, e.g. `"dense_1/kernel:0"`) after being reloaded.
    It is recommended that you use layer attributes to
    access specific variables, e.g. `model.get_layer("dense_1").kernel`.
    """
    is_keras_zip = str(filepath).endswith(".keras") and zipfile.is_zipfile(
        filepath
    )
    is_keras_dir = file_utils.isdir(filepath) and file_utils.exists(
        file_utils.join(filepath, "config.json")
    )

    # Support for remote zip files
    if (
        file_utils.is_remote_path(filepath)
        and not file_utils.isdir(filepath)
        and not is_keras_zip
    ):
        local_path = file_utils.join(
            saving_lib.get_temp_dir(), os.path.basename(filepath)
        )

        # Copy from remote to temporary local directory
        file_utils.copy(filepath, local_path)

        # Switch filepath to local zipfile for loading model
        if zipfile.is_zipfile(local_path):
            filepath = local_path
            is_keras_zip = True

    if is_keras_zip or is_keras_dir:
        return saving_lib.load_model(
            filepath,
            custom_objects=custom_objects,
            compile=compile,
            safe_mode=safe_mode,
        )
    if str(filepath).endswith((".h5", ".hdf5")):
        return legacy_h5_format.load_model_from_hdf5(
            filepath, custom_objects=custom_objects, compile=compile
        )
    elif str(filepath).endswith(".keras"):
        raise ValueError(
            f"File not found: filepath={filepath}. "
            "Please ensure the file is an accessible `.keras` "
            "zip file."
        )
    else:
        raise ValueError(
            f"File format not supported: filepath={filepath}. "
            "Keras 3 only supports V3 `.keras` files and "
            "legacy H5 format files (`.h5` extension). "
            "Note that the legacy SavedModel format is not "
            "supported by `load_model()` in Keras 3. In "
            "order to reload a TensorFlow SavedModel as an "
            "inference-only layer in Keras 3, use "
            "`keras.layers.TFSMLayer("
            f"{filepath}, call_endpoint='serving_default')` "
            "(note that your `call_endpoint` "
            "might have a different name)."
        )


@keras_export("keras.saving.save_weights")
def save_weights(model, filepath, overwrite=True, **kwargs):
    if not str(filepath).endswith(".weights.h5"):
        raise ValueError(
            "The filename must end in `.weights.h5`. "
            f"Received: filepath={filepath}"
        )
    try:
        exists = os.path.exists(filepath)
    except TypeError:
        exists = False
    if exists and not overwrite:
        proceed = io_utils.ask_to_proceed_with_overwrite(filepath)
        if not proceed:
            return
    saving_lib.save_weights_only(model, filepath, **kwargs)


@keras_export("keras.saving.load_weights")
def load_weights(model, filepath, skip_mismatch=False, **kwargs):
    if str(filepath).endswith(".keras"):
        if kwargs:
            raise ValueError(f"Invalid keyword arguments: {kwargs}")
        saving_lib.load_weights_only(
            model, filepath, skip_mismatch=skip_mismatch
        )
    elif str(filepath).endswith(".weights.h5"):
        objects_to_skip = kwargs.pop("objects_to_skip", None)
        if kwargs:
            raise ValueError(f"Invalid keyword arguments: {kwargs}")
        saving_lib.load_weights_only(
            model,
            filepath,
            skip_mismatch=skip_mismatch,
            objects_to_skip=objects_to_skip,
        )
    elif str(filepath).endswith(".h5") or str(filepath).endswith(".hdf5"):
        by_name = kwargs.pop("by_name", False)
        if kwargs:
            raise ValueError(f"Invalid keyword arguments: {kwargs}")
        if not h5py:
            raise ImportError(
                "Loading a H5 file requires `h5py` to be installed."
            )
        with h5py.File(filepath, "r") as f:
            if "layer_names" not in f.attrs and "model_weights" in f:
                f = f["model_weights"]
            if by_name:
                legacy_h5_format.load_weights_from_hdf5_group_by_name(
                    f, model, skip_mismatch
                )
            else:
                legacy_h5_format.load_weights_from_hdf5_group(f, model)
    else:
        raise ValueError(
            f"File format not supported: filepath={filepath}. "
            "Keras 3 only supports V3 `.keras` and `.weights.h5` "
            "files, or legacy V1/V2 `.h5` files."
        )
