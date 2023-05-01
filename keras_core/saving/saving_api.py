import os
import zipfile

from tensorflow.io import gfile

from keras_core.api_export import keras_core_export


@keras_core_export(
    ["keras_core.saving.load_model", "keras_core.models.load_model"]
)
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
            applicable to the Keras v3 model format. Defaults to True.

    Returns:
        A Keras model instance. If the original model was compiled,
        and the argument `compile=True` is set, then the returned model
        will be compiled. Otherwise, the model will be left uncompiled.

    Example:

    ```python
    model = keras_core.Sequential([
        keras_core.layers.Dense(5, input_shape=(3,)),
        keras_core.layers.Softmax()])
    model.save("model.keras")
    loaded_model = keras_core.saving.load_model("model.keras")
    x = np.random.random((10, 3))
    assert np.allclose(model.predict(x), loaded_model.predict(x))
    ```

    Note that the model variables may have different name values
    (`var.name` property, e.g. `"dense_1/kernel:0"`) after being reloaded.
    It is recommended that you use layer attributes to
    access specific variables, e.g. `model.get_layer("dense_1").kernel`.
    """
    from keras_core.saving import saving_lib

    is_keras_zip = str(filepath).endswith(".keras") and zipfile.is_zipfile(
        filepath
    )

    # Support for remote zip files
    if (
        saving_lib.is_remote_path(filepath)
        and not gfile.isdir(filepath)
        and not is_keras_zip
    ):
        local_path = os.path.join(
            saving_lib.get_temp_dir(), os.path.basename(filepath)
        )

        # Copy from remote to temporary local directory
        gfile.copy(filepath, local_path, overwrite=True)

        # Switch filepath to local zipfile for loading model
        if zipfile.is_zipfile(local_path):
            filepath = local_path
            is_keras_zip = True

    if is_keras_zip:
        return saving_lib.load_model(
            filepath,
            custom_objects=custom_objects,
            compile=compile,
            safe_mode=safe_mode,
        )
    else:
        raise ValueError(
            "The file you are trying to load does not appear to "
            "be a valid `.keras` model file. If it is a legacy "
            "`.h5` or SavedModel file, do note that these formats "
            "are not supported in Keras Core."
        )
