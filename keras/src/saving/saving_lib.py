"""Python-based idempotent model-saving functionality."""

import datetime
import io
import json
import pathlib
import tempfile
import warnings
import zipfile

import ml_dtypes
import numpy as np

from keras.src import backend
from keras.src.backend.common import global_state
from keras.src.layers.layer import Layer
from keras.src.losses.loss import Loss
from keras.src.metrics.metric import Metric
from keras.src.optimizers.optimizer import Optimizer
from keras.src.saving.serialization_lib import ObjectSharingScope
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object
from keras.src.trainers.compile_utils import CompileMetrics
from keras.src.utils import file_utils
from keras.src.utils import io_utils
from keras.src.utils import naming
from keras.src.utils import plot_model
from keras.src.utils.model_visualization import check_pydot
from keras.src.utils.summary_utils import weight_memory_size
from keras.src.version import __version__ as keras_version

try:
    import h5py
except ImportError:
    h5py = None
try:
    import psutil
except ImportError:
    psutil = None
try:
    import huggingface_hub
except ImportError:
    huggingface_hub = None


_CONFIG_FILENAME = "config.json"
_METADATA_FILENAME = "metadata.json"
_VARS_FNAME = "model.weights"  # Will become e.g. "model.weights.h5"
_VARS_FNAME_H5 = _VARS_FNAME + ".h5"
_VARS_FNAME_NPZ = _VARS_FNAME + ".npz"
_ASSETS_DIRNAME = "assets"
_MEMORY_UPPER_BOUND = 0.5  # 50%


_MODEL_CARD_TEMPLATE = """
---
library_name: keras
---

This model has been uploaded using the Keras library and can be used with JAX,
TensorFlow, and PyTorch backends.

This model card has been generated automatically and should be completed by the
model author.
See [Model Cards documentation](https://huggingface.co/docs/hub/model-cards) for
more information.

For more details about the model architecture, check out
[config.json](./config.json)."""


def save_model(model, filepath, weights_format="h5", zipped=True):
    """Save a zip-archive representing a Keras model to the given file or path.

    The zip-based archive contains the following structure:

    - JSON-based configuration file (config.json): Records of model, layer, and
        other saveables' configuration.
    - H5-based saveable state files, found in respective directories, such as
        model/states.npz, model/dense_layer/states.npz, etc.
    - Metadata file.

    The states of Keras saveables (layers, optimizers, loss, and metrics) are
    automatically saved as long as they can be discovered through the attributes
    returned by `dir(Model)`. Typically, the state includes the variables
    associated with the saveable, but some specially purposed layers may
    contain more such as the vocabularies stored in the hashmaps. The saveables
    define how their states are saved by exposing `save_state()` and
    `load_state()` APIs.

    For the case of layer states, the variables will be visited as long as
    they are either 1) referenced via layer attributes, or 2) referenced via a
    container (list, tuple, or dict), and the container is referenced via a
    layer attribute.
    """
    if weights_format == "h5" and h5py is None:
        raise ImportError("h5py must be installed in order to save a model.")

    if not model.built:
        warnings.warn(
            "You are saving a model that has not yet been built. "
            "It might not contain any weights yet. "
            "Consider building the model first by calling it "
            "on some data.",
            stacklevel=2,
        )

    if isinstance(filepath, io.IOBase):
        _save_model_to_fileobj(model, filepath, weights_format)
        return

    filepath = str(filepath)
    is_hf = filepath.startswith("hf://")
    if zipped and not filepath.endswith(".keras"):
        raise ValueError(
            "Invalid `filepath` argument: expected a `.keras` extension. "
            f"Received: filepath={filepath}"
        )
    if not zipped and filepath.endswith(".keras"):
        raise ValueError(
            "When using `zipped=False`, the `filepath` argument should not "
            f"end in `.keras`. Received: filepath={filepath}"
        )
    if zipped and is_hf:
        raise ValueError(
            "When saving to the Hugging Face Hub, you should not save the "
            f"model as zipped. Received: filepath={filepath}, zipped={zipped}"
        )
    if is_hf:
        _upload_model_to_hf(model, filepath, weights_format)
    elif not zipped:
        _save_model_to_dir(model, filepath, weights_format)
    else:
        if file_utils.is_remote_path(filepath):
            # Remote path. Zip to local memory byte io and copy to remote
            zip_filepath = io.BytesIO()
            _save_model_to_fileobj(model, zip_filepath, weights_format)
            with file_utils.File(filepath, "wb") as f:
                f.write(zip_filepath.getvalue())
        else:
            with open(filepath, "wb") as f:
                _save_model_to_fileobj(model, f, weights_format)


def _serialize_model_as_json(model):
    with ObjectSharingScope():
        serialized_model_dict = serialize_keras_object(model)
    config_json = json.dumps(serialized_model_dict)
    metadata_json = json.dumps(
        {
            "keras_version": keras_version,
            "date_saved": datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S"),
        }
    )
    return config_json, metadata_json


def _save_model_to_dir(model, dirpath, weights_format):
    if not file_utils.exists(dirpath):
        file_utils.makedirs(dirpath)
    config_json, metadata_json = _serialize_model_as_json(model)
    with open(file_utils.join(dirpath, _METADATA_FILENAME), "w") as f:
        f.write(metadata_json)
    with open(file_utils.join(dirpath, _CONFIG_FILENAME), "w") as f:
        f.write(config_json)
    weights_filepath = file_utils.join(dirpath, _VARS_FNAME_H5)
    assert_dirpath = file_utils.join(dirpath, _ASSETS_DIRNAME)
    try:
        if weights_format == "h5":
            weights_store = H5IOStore(weights_filepath, mode="w")
        elif weights_format == "npz":
            weights_store = NpzIOStore(weights_filepath, mode="w")
        else:
            raise ValueError(
                "Unknown `weights_format` argument. "
                "Expected 'h5' or 'npz'. "
                f"Received: weights_format={weights_format}"
            )
        asset_store = DiskIOStore(assert_dirpath, mode="w")
        _save_state(
            model,
            weights_store=weights_store,
            assets_store=asset_store,
            inner_path="",
            visited_saveables=set(),
        )
    finally:
        weights_store.close()
        asset_store.close()


def _save_model_to_fileobj(model, fileobj, weights_format):
    config_json, metadata_json = _serialize_model_as_json(model)

    with zipfile.ZipFile(fileobj, "w") as zf:
        with zf.open(_METADATA_FILENAME, "w") as f:
            f.write(metadata_json.encode())
        with zf.open(_CONFIG_FILENAME, "w") as f:
            f.write(config_json.encode())

        weights_file_path = None
        weights_store = None
        asset_store = None
        write_zf = False
        try:
            if weights_format == "h5":
                try:
                    if is_memory_sufficient(model):
                        # Load the model weights into memory before writing
                        # .keras if the system memory is sufficient.
                        weights_store = H5IOStore(
                            _VARS_FNAME_H5, archive=zf, mode="w"
                        )
                    else:
                        # Try opening the .h5 file, then writing it to `zf` at
                        # the end of the function call. This is more memory
                        # efficient than writing the weights into memory first.
                        working_dir = pathlib.Path(fileobj.name).parent
                        weights_file_path = tempfile.NamedTemporaryFile(
                            dir=working_dir
                        )
                        weights_store = H5IOStore(
                            weights_file_path.name, mode="w"
                        )
                        write_zf = True
                except:
                    # If we can't use the local disk for any reason, write the
                    # weights into memory first, which consumes more memory.
                    weights_store = H5IOStore(
                        _VARS_FNAME_H5, archive=zf, mode="w"
                    )
            elif weights_format == "npz":
                weights_store = NpzIOStore(
                    _VARS_FNAME_NPZ, archive=zf, mode="w"
                )
            else:
                raise ValueError(
                    "Unknown `weights_format` argument. "
                    "Expected 'h5' or 'npz'. "
                    f"Received: weights_format={weights_format}"
                )

            asset_store = DiskIOStore(_ASSETS_DIRNAME, archive=zf, mode="w")

            _save_state(
                model,
                weights_store=weights_store,
                assets_store=asset_store,
                inner_path="",
                visited_saveables=set(),
            )
        except:
            # Skip the final `zf.write` if any exception is raised
            write_zf = False
            if weights_store:
                weights_store.archive = None
            raise
        finally:
            if weights_store:
                weights_store.close()
            if asset_store:
                asset_store.close()
            if write_zf and weights_file_path:
                zf.write(weights_file_path.name, _VARS_FNAME_H5)
            if weights_file_path:
                weights_file_path.close()


def _upload_model_to_hf(model, hf_path, weights_format):
    if huggingface_hub is None:
        raise ImportError(
            "To save models to the Hugging Face Hub, "
            "you must install the `huggingface_hub` package."
        )

    original_hf_path = hf_path
    if hf_path.startswith("hf://"):
        hf_path = hf_path[5:]
    if hf_path.count("/") > 1:
        raise ValueError(
            "Invalid `hf_path` argument: expected `namespace/model_name`"
            f" format. Received: hf_path={original_hf_path}"
        )

    api = huggingface_hub.HfApi(
        library_name="keras", library_version=keras_version
    )
    repo_url = api.create_repo(hf_path, exist_ok=True)
    repo_id = repo_url.repo_id

    with tempfile.TemporaryDirectory() as tmp_dir:
        _save_model_to_dir(model, tmp_dir, weights_format)

        model_card = _MODEL_CARD_TEMPLATE

        if check_pydot():
            plot_path = file_utils.join(tmp_dir, "assets", "summary_plot.png")
            plot_model(
                model,
                to_file=plot_path,
                show_layer_names=True,
                show_shapes=True,
                show_dtype=True,
            )
            if len(model.layers) <= 10:
                model_card += "\n\n![](./assets/summary_plot.png)"
            else:
                model_card += (
                    "A plot of the model can be found "
                    "[here](./assets/summary_plot.png)."
                )

        with open(file_utils.join(tmp_dir, "README.md"), "w") as f:
            f.write(model_card)

        api.upload_folder(
            repo_id=repo_id,
            folder_path=tmp_dir,
            commit_message="Save model using Keras.",
        )
        io_utils.print_msg(
            f"Model saved to the Hugging Face Hub: {repo_url}\n"
            "To load back the model, use "
            f"`keras.saving.load_model('hf://{repo_id}')`"
        )


def load_model(filepath, custom_objects=None, compile=True, safe_mode=True):
    """Load a zip archive representing a Keras model."""
    if isinstance(filepath, io.IOBase):
        return _load_model_from_fileobj(
            filepath, custom_objects, compile, safe_mode
        )
    elif str(filepath).startswith("hf://"):
        if huggingface_hub is None:
            raise ImportError(
                "To load models from the Hugging Face Hub, "
                "you must install the `huggingface_hub` package."
            )

        repo_id = filepath[5:]
        folder_path = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            library_name="keras",
            library_version=keras_version,
        )
        return _load_model_from_dir(
            folder_path, custom_objects, compile, safe_mode
        )
    else:
        filepath = str(filepath)
        if not filepath.endswith(".keras"):
            is_keras_dir = file_utils.isdir(filepath) and file_utils.exists(
                file_utils.join(filepath, "config.json")
            )
            if is_keras_dir:
                return _load_model_from_dir(
                    filepath, custom_objects, compile, safe_mode
                )
            raise ValueError(
                "Invalid filename: expected a `.keras` extension. "
                f"Received: filepath={filepath}"
            )
        with open(filepath, "rb") as f:
            return _load_model_from_fileobj(
                f, custom_objects, compile, safe_mode
            )


def _load_model_from_dir(dirpath, custom_objects, compile, safe_mode):
    if not file_utils.exists(dirpath):
        raise ValueError(f"Directory doesn't exist: {dirpath}")
    if not file_utils.isdir(dirpath):
        raise ValueError(f"Path isn't a directory: {dirpath}")

    with open(file_utils.join(dirpath, _CONFIG_FILENAME), "r") as f:
        config_json = f.read()
    model = _model_from_config(config_json, custom_objects, compile, safe_mode)

    all_filenames = file_utils.listdir(dirpath)
    try:
        if _VARS_FNAME_H5 in all_filenames:
            weights_file_path = file_utils.join(dirpath, _VARS_FNAME_H5)
            weights_store = H5IOStore(weights_file_path, mode="r")
        elif _VARS_FNAME_NPZ in all_filenames:
            weights_file_path = file_utils.join(dirpath, _VARS_FNAME_NPZ)
            weights_store = NpzIOStore(weights_file_path, mode="r")
        else:
            raise ValueError(
                f"Expected a {_VARS_FNAME_H5} or {_VARS_FNAME_NPZ} file."
            )
        if len(all_filenames) > 3:
            asset_store = DiskIOStore(
                file_utils.join(dirpath, _ASSETS_DIRNAME), mode="r"
            )

        else:
            asset_store = None

        failed_saveables = set()
        error_msgs = {}
        _load_state(
            model,
            weights_store=weights_store,
            assets_store=asset_store,
            inner_path="",
            visited_saveables=set(),
            failed_saveables=failed_saveables,
            error_msgs=error_msgs,
        )

    finally:
        weights_store.close()
        if asset_store:
            asset_store.close()

    if failed_saveables:
        _raise_loading_failure(error_msgs)
    return model


def _model_from_config(config_json, custom_objects, compile, safe_mode):
    # Note: we should NOT use a custom JSON decoder. Anything that
    # needs custom decoding must be handled in deserialize_keras_object.
    config_dict = json.loads(config_json)
    if not compile:
        # Disable compilation
        config_dict["compile_config"] = None
    # Construct the model from the configuration file in the archive.
    with ObjectSharingScope():
        model = deserialize_keras_object(
            config_dict, custom_objects, safe_mode=safe_mode
        )
    return model


def _load_model_from_fileobj(fileobj, custom_objects, compile, safe_mode):
    with zipfile.ZipFile(fileobj, "r") as zf:
        with zf.open(_CONFIG_FILENAME, "r") as f:
            config_json = f.read()

        model = _model_from_config(
            config_json, custom_objects, compile, safe_mode
        )

        all_filenames = zf.namelist()
        extract_dir = None
        weights_store = None
        asset_store = None
        try:
            if _VARS_FNAME_H5 in all_filenames:
                try:
                    if is_memory_sufficient(model):
                        # Load the entire file into memory if the system memory
                        # is sufficient.
                        io_file = io.BytesIO(
                            zf.open(_VARS_FNAME_H5, "r").read()
                        )
                        weights_store = H5IOStore(io_file, mode="r")
                    else:
                        # Try extracting the model.weights.h5 file, and then
                        # loading it using using h5py. This is significantly
                        # faster than reading from the zip archive on the fly.
                        extract_dir = tempfile.TemporaryDirectory(
                            dir=pathlib.Path(fileobj.name).parent
                        )
                        zf.extract(_VARS_FNAME_H5, extract_dir.name)
                        weights_store = H5IOStore(
                            pathlib.Path(extract_dir.name, _VARS_FNAME_H5),
                            mode="r",
                        )
                except:
                    # If we can't use the local disk for any reason, read the
                    # weights from the zip archive on the fly, which is less
                    # efficient.
                    weights_store = H5IOStore(_VARS_FNAME_H5, zf, mode="r")
            elif _VARS_FNAME_NPZ in all_filenames:
                weights_store = NpzIOStore(_VARS_FNAME_NPZ, zf, mode="r")
            else:
                raise ValueError(
                    f"Expected a {_VARS_FNAME_H5} or {_VARS_FNAME_NPZ} file."
                )

            if len(all_filenames) > 3:
                asset_store = DiskIOStore(_ASSETS_DIRNAME, archive=zf, mode="r")

            failed_saveables = set()
            error_msgs = {}
            _load_state(
                model,
                weights_store=weights_store,
                assets_store=asset_store,
                inner_path="",
                visited_saveables=set(),
                failed_saveables=failed_saveables,
                error_msgs=error_msgs,
            )
        finally:
            if weights_store:
                weights_store.close()
            if asset_store:
                asset_store.close()
            if extract_dir:
                extract_dir.cleanup()

        if failed_saveables:
            _raise_loading_failure(error_msgs)
    return model


def save_weights_only(model, filepath, objects_to_skip=None):
    """Save only the weights of a model to a target filepath (.weights.h5).

    Note: only supports h5 for now.
    """
    # TODO: if h5 filepath is remote, create the file in a temporary directory
    # then upload it
    filepath = str(filepath)
    if not filepath.endswith(".weights.h5"):
        raise ValueError(
            "Invalid `filepath` argument: expected a `.weights.h5` extension. "
            f"Received: filepath={filepath}"
        )
    weights_store = H5IOStore(filepath, mode="w")
    if objects_to_skip is not None:
        visited_saveables = set(id(o) for o in objects_to_skip)
    else:
        visited_saveables = set()
    _save_state(
        model,
        weights_store=weights_store,
        assets_store=None,
        inner_path="",
        visited_saveables=visited_saveables,
    )
    weights_store.close()


def load_weights_only(
    model, filepath, skip_mismatch=False, objects_to_skip=None
):
    """Load the weights of a model from a filepath (.keras or .weights.h5).

    Note: only supports h5 for now.
    """
    archive = None
    filepath = str(filepath)
    if filepath.endswith(".weights.h5"):
        # TODO: download file if h5 filepath is remote
        weights_store = H5IOStore(filepath, mode="r")
    elif filepath.endswith(".keras"):
        archive = zipfile.ZipFile(filepath, "r")
        weights_store = H5IOStore(_VARS_FNAME_H5, archive=archive, mode="r")

    failed_saveables = set()
    if objects_to_skip is not None:
        visited_saveables = set(id(o) for o in objects_to_skip)
    else:
        visited_saveables = set()
    error_msgs = {}
    _load_state(
        model,
        weights_store=weights_store,
        assets_store=None,
        inner_path="",
        skip_mismatch=skip_mismatch,
        visited_saveables=visited_saveables,
        failed_saveables=failed_saveables,
        error_msgs=error_msgs,
    )
    weights_store.close()
    if archive:
        archive.close()

    if failed_saveables:
        _raise_loading_failure(error_msgs, warn_only=skip_mismatch)


def _raise_loading_failure(error_msgs, warn_only=False):
    first_key = list(error_msgs.keys())[0]
    ex_saveable, ex_error = error_msgs[first_key]
    msg = (
        f"A total of {len(error_msgs)} objects could not "
        "be loaded. Example error message for "
        f"object {ex_saveable}:\n\n"
        f"{ex_error}\n\n"
        "List of objects that could not be loaded:\n"
        f"{[x[0] for x in error_msgs.values()]}"
    )
    if warn_only:
        warnings.warn(msg)
    else:
        raise ValueError(msg)


def _write_to_zip_recursively(zipfile_to_save, system_path, zip_path):
    if not file_utils.isdir(system_path):
        zipfile_to_save.write(system_path, zip_path)
    else:
        for file_name in file_utils.listdir(system_path):
            system_file_path = file_utils.join(system_path, file_name).replace(
                "\\", "/"
            )
            zip_file_path = file_utils.join(zip_path, file_name).replace(
                "\\", "/"
            )
            _write_to_zip_recursively(
                zipfile_to_save, system_file_path, zip_file_path
            )


def _name_key(name):
    """Make sure that private attributes are visited last."""
    if name.startswith("_"):
        return "~" + name
    return name


def _walk_saveable(saveable):
    from keras.src.saving.keras_saveable import KerasSaveable

    if not isinstance(saveable, KerasSaveable):
        raise ValueError(
            "Expected object to be an "
            "instance of `KerasSaveable`, but "
            f"got {saveable} of type {type(saveable)}"
        )

    obj_type = saveable._obj_type()
    attr_skipset = get_attr_skipset(obj_type)

    # Save all layers directly tracked by Sequential and Functional first.
    # This helps avoid ordering concerns for subclassed Sequential or Functional
    # models with extra attributes--the internal Keras state take precedence.
    if obj_type in ("Sequential", "Functional"):
        yield "layers", saveable.layers

    for child_attr in sorted(dir(saveable), key=lambda x: _name_key(x)):
        if child_attr.startswith("__") or child_attr in attr_skipset:
            continue
        try:
            child_obj = getattr(saveable, child_attr)
        except Exception:
            # Avoid raising the exception when visiting the attributes.
            continue
        yield child_attr, child_obj


def _save_state(
    saveable,
    weights_store,
    assets_store,
    inner_path,
    visited_saveables,
):
    from keras.src.saving.keras_saveable import KerasSaveable

    # If the saveable has already been saved, skip it.
    if id(saveable) in visited_saveables:
        return

    if hasattr(saveable, "save_own_variables") and weights_store:
        if hasattr(saveable, "name") and isinstance(saveable.name, str):
            metadata = {"name": saveable.name}
        else:
            metadata = None
        saveable.save_own_variables(
            weights_store.make(inner_path, metadata=metadata)
        )
    if hasattr(saveable, "save_assets") and assets_store:
        saveable.save_assets(assets_store.make(inner_path))

    visited_saveables.add(id(saveable))

    # Recursively save state of children saveables (layers, optimizers, etc.)
    for child_attr, child_obj in _walk_saveable(saveable):
        if isinstance(child_obj, KerasSaveable):
            _save_state(
                child_obj,
                weights_store,
                assets_store,
                inner_path=file_utils.join(inner_path, child_attr).replace(
                    "\\", "/"
                ),
                visited_saveables=visited_saveables,
            )
        elif isinstance(child_obj, (list, dict, tuple, set)):
            _save_container_state(
                child_obj,
                weights_store,
                assets_store,
                inner_path=file_utils.join(inner_path, child_attr).replace(
                    "\\", "/"
                ),
                visited_saveables=visited_saveables,
            )


def _load_state(
    saveable,
    weights_store,
    assets_store,
    inner_path,
    skip_mismatch=False,
    visited_saveables=None,
    failed_saveables=None,
    error_msgs=None,
):
    from keras.src.saving.keras_saveable import KerasSaveable

    if visited_saveables and id(saveable) in visited_saveables:
        return

    failure = False

    if hasattr(saveable, "load_own_variables") and weights_store:
        if skip_mismatch or failed_saveables is not None:
            try:
                saveable.load_own_variables(weights_store.get(inner_path))
            except Exception as e:
                failed_saveables.add(id(saveable))
                error_msgs[id(saveable)] = saveable, e
                failure = True
        else:
            saveable.load_own_variables(weights_store.get(inner_path))

    if hasattr(saveable, "load_assets") and assets_store:
        if skip_mismatch or failed_saveables is not None:
            try:
                saveable.load_assets(assets_store.get(inner_path))
            except Exception as e:
                failed_saveables.add(id(saveable))
                error_msgs[id(saveable)] = saveable, e
                failure = True
        else:
            saveable.load_assets(assets_store.get(inner_path))

    if failed_saveables is not None:
        currently_failed = len(failed_saveables)
    else:
        currently_failed = 0

    # Recursively load states for Keras saveables such as layers/optimizers.
    for child_attr, child_obj in _walk_saveable(saveable):
        if isinstance(child_obj, KerasSaveable):
            _load_state(
                child_obj,
                weights_store,
                assets_store,
                inner_path=file_utils.join(inner_path, child_attr).replace(
                    "\\", "/"
                ),
                skip_mismatch=skip_mismatch,
                visited_saveables=visited_saveables,
                failed_saveables=failed_saveables,
                error_msgs=error_msgs,
            )
        elif isinstance(child_obj, (list, dict, tuple, set)):
            _load_container_state(
                child_obj,
                weights_store,
                assets_store,
                inner_path=file_utils.join(inner_path, child_attr).replace(
                    "\\", "/"
                ),
                skip_mismatch=skip_mismatch,
                visited_saveables=visited_saveables,
                failed_saveables=failed_saveables,
                error_msgs=error_msgs,
            )

    if failed_saveables is not None:
        newly_failed = len(failed_saveables) - currently_failed
    else:
        newly_failed = 0

    if not failure:
        if visited_saveables is not None and newly_failed <= 0:
            visited_saveables.add(id(saveable))
        if id(saveable) in failed_saveables:
            failed_saveables.remove(id(saveable))
            error_msgs.pop(id(saveable))


def _save_container_state(
    container, weights_store, assets_store, inner_path, visited_saveables
):
    from keras.src.saving.keras_saveable import KerasSaveable

    used_names = {}
    if isinstance(container, dict):
        container = list(container.values())

    for saveable in container:
        if isinstance(saveable, KerasSaveable):
            # Do NOT address the saveable via `saveable.name`, since
            # names are usually autogenerated and thus not reproducible
            # (i.e. they may vary across two instances of the same model).
            name = naming.to_snake_case(saveable.__class__.__name__)
            if name in used_names:
                used_names[name] += 1
                name = f"{name}_{used_names[name]}"
            else:
                used_names[name] = 0
            _save_state(
                saveable,
                weights_store,
                assets_store,
                inner_path=file_utils.join(inner_path, name).replace("\\", "/"),
                visited_saveables=visited_saveables,
            )


def _load_container_state(
    container,
    weights_store,
    assets_store,
    inner_path,
    skip_mismatch,
    visited_saveables,
    failed_saveables,
    error_msgs,
):
    from keras.src.saving.keras_saveable import KerasSaveable

    used_names = {}
    if isinstance(container, dict):
        container = list(container.values())

    for saveable in container:
        if isinstance(saveable, KerasSaveable):
            name = naming.to_snake_case(saveable.__class__.__name__)
            if name in used_names:
                used_names[name] += 1
                name = f"{name}_{used_names[name]}"
            else:
                used_names[name] = 0
            _load_state(
                saveable,
                weights_store,
                assets_store,
                inner_path=file_utils.join(inner_path, name).replace("\\", "/"),
                skip_mismatch=skip_mismatch,
                visited_saveables=visited_saveables,
                failed_saveables=failed_saveables,
                error_msgs=error_msgs,
            )


class DiskIOStore:
    """Asset store backed by disk storage.

    If `archive` is specified, then `root_path` refers to the filename
    inside the archive.

    If `archive` is not specified, then `root_path` refers to the full path of
    the target directory.
    """

    def __init__(self, root_path, archive=None, mode=None):
        self.mode = mode
        self.root_path = root_path
        self.archive = archive
        self.tmp_dir = None
        if self.archive:
            self.tmp_dir = get_temp_dir()
            if self.mode == "r":
                self.archive.extractall(path=self.tmp_dir)
            self.working_dir = file_utils.join(
                self.tmp_dir, self.root_path
            ).replace("\\", "/")
            if self.mode == "w":
                file_utils.makedirs(self.working_dir)
        else:
            if mode == "r":
                self.working_dir = root_path
            else:
                self.tmp_dir = get_temp_dir()
                self.working_dir = file_utils.join(
                    self.tmp_dir, self.root_path
                ).replace("\\", "/")
                file_utils.makedirs(self.working_dir)

    def make(self, path):
        if not path:
            return self.working_dir
        path = file_utils.join(self.working_dir, path).replace("\\", "/")
        if not file_utils.exists(path):
            file_utils.makedirs(path)
        return path

    def get(self, path):
        if not path:
            return self.working_dir
        path = file_utils.join(self.working_dir, path).replace("\\", "/")
        if file_utils.exists(path):
            return path
        return None

    def close(self):
        if self.mode == "w" and self.archive:
            _write_to_zip_recursively(
                self.archive, self.working_dir, self.root_path
            )
        if self.tmp_dir and file_utils.exists(self.tmp_dir):
            file_utils.rmtree(self.tmp_dir)


class H5IOStore:
    def __init__(self, root_path, archive=None, mode="r"):
        """Numerical variable store backed by HDF5.

        If `archive` is specified, then `root_path` refers to the filename
        inside the archive.

        If `archive` is not specified, then `root_path` refers to the path of
        the h5 file on disk.
        """
        self.root_path = root_path
        self.mode = mode
        self.archive = archive
        self.io_file = None

        if self.archive:
            if self.mode == "w":
                self.io_file = io.BytesIO()
            else:
                self.io_file = self.archive.open(self.root_path, "r")
            self.h5_file = h5py.File(self.io_file, mode=self.mode)
        else:
            self.h5_file = h5py.File(root_path, mode=self.mode)

    def make(self, path, metadata=None):
        return H5Entry(self.h5_file, path, mode="w", metadata=metadata)

    def get(self, path):
        return H5Entry(self.h5_file, path, mode="r")

    def close(self):
        self.h5_file.close()
        if self.mode == "w" and self.archive:
            self.archive.writestr(self.root_path, self.io_file.getvalue())
        if self.io_file:
            self.io_file.close()


class H5Entry:
    """Leaf entry in a H5IOStore."""

    def __init__(self, h5_file, path, mode, metadata=None):
        self.h5_file = h5_file
        self.path = path
        self.mode = mode
        self.metadata = metadata

        if mode == "w":
            if not path:
                self.group = self.h5_file.create_group("vars")
            else:
                self.group = self.h5_file.create_group(self.path).create_group(
                    "vars"
                )
            if self.metadata:
                for k, v in self.metadata.items():
                    self.group.attrs[k] = v
        else:
            found = False
            if not path:
                if "vars" in self.h5_file:
                    self.group = self.h5_file["vars"]
                    found = True
            elif path in self.h5_file and "vars" in self.h5_file[path]:
                self.group = self.h5_file[path]["vars"]
                found = True
            else:
                # No hit.
                # Fix for 2.13 compatibility
                if "_layer_checkpoint_dependencies" in self.h5_file:
                    path = path.replace(
                        "layers", "_layer_checkpoint_dependencies"
                    )
                    self.path = path
                    if path in self.h5_file and "vars" in self.h5_file[path]:
                        self.group = self.h5_file[path]["vars"]
                        found = True
            if not found:
                self.group = {}

    def __len__(self):
        return self.group.__len__()

    def keys(self):
        return self.group.keys()

    def items(self):
        return self.group.items()

    def values(self):
        return self.group.values()

    def __setitem__(self, key, value):
        if self.mode != "w":
            raise ValueError("Setting a value is only allowed in write mode.")
        value = backend.convert_to_numpy(value)
        if backend.standardize_dtype(value.dtype) == "bfloat16":
            ds = self.group.create_dataset(key, data=value)
            ds.attrs["dtype"] = "bfloat16"
        else:
            self.group[key] = value

    def __getitem__(self, name):
        value = self.group[name]
        if "dtype" in value.attrs and value.attrs["dtype"] == "bfloat16":
            value = np.array(value, dtype=ml_dtypes.bfloat16)
        return value


class NpzIOStore:
    def __init__(self, root_path, archive=None, mode="r"):
        """Numerical variable store backed by NumPy.savez/load.

         If `archive` is specified, then `root_path` refers to the filename
        inside the archive.

        If `archive` is not specified, then `root_path` refers to the path of
        the npz file on disk.
        """
        self.root_path = root_path
        self.mode = mode
        self.archive = archive
        if mode == "w":
            self.contents = {}
        else:
            if self.archive:
                self.f = archive.open(root_path, mode="r")
            else:
                self.f = open(root_path, mode="rb")
            self.contents = np.load(self.f, allow_pickle=True)

    def make(self, path, metadata=None):
        if not path:
            self.contents["__root__"] = {}
            return self.contents["__root__"]
        self.contents[path] = {}
        return self.contents[path]

    def get(self, path):
        if not path:
            if "__root__" in self.contents:
                return dict(self.contents["__root__"])
            return {}
        if path in self.contents:
            return self.contents[path].tolist()
        return {}

    def close(self):
        if self.mode == "w":
            if self.archive:
                self.f = self.archive.open(
                    self.root_path, mode="w", force_zip64=True
                )
            else:
                self.f = open(self.root_path, mode="wb")
            np.savez(self.f, **self.contents)
        self.f.close()


def get_temp_dir():
    temp_dir = tempfile.mkdtemp()
    testfile = tempfile.TemporaryFile(dir=temp_dir)
    testfile.close()
    return temp_dir


def get_attr_skipset(obj_type):
    skipset = global_state.get_global_attribute(
        f"saving_attr_skiplist_{obj_type}", None
    )
    if skipset is not None:
        return skipset

    skipset = set(
        [
            "_self_unconditional_dependency_names",
        ]
    )
    if obj_type == "Layer":
        ref_obj = Layer()
        skipset.update(dir(ref_obj))
    elif obj_type == "Functional":
        ref_obj = Layer()
        skipset.update(dir(ref_obj) + ["operations", "_operations"])
    elif obj_type == "Sequential":
        ref_obj = Layer()
        skipset.update(dir(ref_obj) + ["_functional"])
    elif obj_type == "Metric":
        ref_obj_a = Metric()
        ref_obj_b = CompileMetrics([], [])
        skipset.update(dir(ref_obj_a) + dir(ref_obj_b))
    elif obj_type == "Optimizer":
        ref_obj = Optimizer(1.0)
        skipset.update(dir(ref_obj))
        skipset.remove("variables")
    elif obj_type == "Loss":
        ref_obj = Loss()
        skipset.update(dir(ref_obj))
    else:
        raise ValueError(
            f"get_attr_skipset got invalid {obj_type=}. "
            "Accepted values for `obj_type` are "
            "['Layer', 'Functional', 'Sequential', 'Metric', "
            "'Optimizer', 'Loss']"
        )

    global_state.set_global_attribute(
        f"saving_attr_skipset_{obj_type}", skipset
    )
    return skipset


def is_memory_sufficient(model):
    """Check if there is sufficient memory to load the model into memory.

    If psutil is installed, we can use it to determine whether the memory is
    sufficient. Otherwise, we use a predefined value of 1 GB for available
    memory.
    """
    if psutil is None:
        available_memory = 1024 * 1024 * 1024  # 1 GB in bytes
    else:
        available_memory = psutil.virtual_memory().available  # In bytes
    return (
        weight_memory_size(model.variables)
        < available_memory * _MEMORY_UPPER_BOUND
    )
