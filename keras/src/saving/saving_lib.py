"""Python-based idempotent model-saving functionality."""

import datetime
import io
import json
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
from keras.src.utils import naming
from keras.src.version import __version__ as keras_version

try:
    import h5py
except ImportError:
    h5py = None

_CONFIG_FILENAME = "config.json"
_METADATA_FILENAME = "metadata.json"
_VARS_FNAME = "model.weights"  # Will become e.g. "model.weights.h5"
_ASSETS_DIRNAME = "assets"


def save_model(model, filepath, weights_format="h5"):
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
    if not filepath.endswith(".keras"):
        raise ValueError(
            "Invalid `filepath` argument: expected a `.keras` extension. "
            f"Received: filepath={filepath}"
        )
    if file_utils.is_remote_path(filepath):
        # Remote path. Zip to local memory byte io and copy to remote
        zip_filepath = io.BytesIO()
        _save_model_to_fileobj(model, zip_filepath, weights_format)
        with file_utils.File(filepath, "wb") as f:
            f.write(zip_filepath.getvalue())
    else:
        with open(filepath, "wb") as f:
            _save_model_to_fileobj(model, f, weights_format)


def _save_model_to_fileobj(model, fileobj, weights_format):
    with ObjectSharingScope():
        serialized_model_dict = serialize_keras_object(model)
    config_json = json.dumps(serialized_model_dict)
    metadata_json = json.dumps(
        {
            "keras_version": keras_version,
            "date_saved": datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S"),
        }
    )

    with zipfile.ZipFile(fileobj, "w") as zf:
        with zf.open(_METADATA_FILENAME, "w") as f:
            f.write(metadata_json.encode())
        with zf.open(_CONFIG_FILENAME, "w") as f:
            f.write(config_json.encode())

        if weights_format == "h5":
            weights_store = H5IOStore(_VARS_FNAME + ".h5", archive=zf, mode="w")
        elif weights_format == "npz":
            weights_store = NpzIOStore(
                _VARS_FNAME + ".npz", archive=zf, mode="w"
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
        weights_store.close()
        asset_store.close()


def load_model(filepath, custom_objects=None, compile=True, safe_mode=True):
    """Load a zip archive representing a Keras model."""
    if isinstance(filepath, io.IOBase):
        return _load_model_from_fileobj(
            filepath, custom_objects, compile, safe_mode
        )
    else:
        filepath = str(filepath)
        if not filepath.endswith(".keras"):
            raise ValueError(
                "Invalid filename: expected a `.keras` extension. "
                f"Received: filepath={filepath}"
            )
        with open(filepath, "rb") as f:
            return _load_model_from_fileobj(
                f, custom_objects, compile, safe_mode
            )


def _load_model_from_fileobj(fileobj, custom_objects, compile, safe_mode):
    with zipfile.ZipFile(fileobj, "r") as zf:
        with zf.open(_CONFIG_FILENAME, "r") as f:
            config_json = f.read()

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

        all_filenames = zf.namelist()
        if _VARS_FNAME + ".h5" in all_filenames:
            weights_store = H5IOStore(_VARS_FNAME + ".h5", archive=zf, mode="r")
        elif _VARS_FNAME + ".npz" in all_filenames:
            weights_store = NpzIOStore(
                _VARS_FNAME + ".npz", archive=zf, mode="r"
            )
        else:
            raise ValueError(
                f"Expected a {_VARS_FNAME}.h5 or {_VARS_FNAME}.npz file."
            )

        if len(all_filenames) > 3:
            asset_store = DiskIOStore(_ASSETS_DIRNAME, archive=zf, mode="r")
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
        weights_store.close()
        if asset_store:
            asset_store.close()

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
        weights_store = H5IOStore(
            _VARS_FNAME + ".h5", archive=archive, mode="r"
        )

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
    attr_skiplist = get_attr_skiplist(obj_type)

    # Save all layers directly tracked by Sequential and Functional first.
    # This helps avoid ordering concerns for subclassed Sequential or Functional
    # models with extra attributes--the internal Keras state take precedence.
    if obj_type in ("Sequential", "Functional"):
        yield "layers", saveable.layers

    for child_attr in sorted(dir(saveable), key=lambda x: _name_key(x)):
        if child_attr.startswith("__") or child_attr in attr_skiplist:
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
        saveable.save_own_variables(weights_store.make(inner_path))
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

    def make(self, path):
        return H5Entry(self.h5_file, path, mode="w")

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

    def __init__(self, h5_file, path, mode):
        self.h5_file = h5_file
        self.path = path
        self.mode = mode

        if mode == "w":
            if not path:
                self.group = self.h5_file.create_group("vars")
            else:
                self.group = self.h5_file.create_group(self.path).create_group(
                    "vars"
                )
        else:
            found = False
            if not path:
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

    def make(self, path):
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


def get_attr_skiplist(obj_type):
    skiplist = global_state.get_global_attribute(
        f"saving_attr_skiplist_{obj_type}", None
    )
    if skiplist is not None:
        return skiplist

    skiplist = [
        "_self_unconditional_dependency_names",
    ]
    if obj_type == "Layer":
        ref_obj = Layer()
        skiplist += dir(ref_obj)
    elif obj_type == "Functional":
        ref_obj = Layer()
        skiplist += dir(ref_obj) + ["operations", "_operations"]
    elif obj_type == "Sequential":
        ref_obj = Layer()
        skiplist += dir(ref_obj) + ["_functional"]
    elif obj_type == "Metric":
        ref_obj_a = Metric()
        ref_obj_b = CompileMetrics([], [])
        skiplist += dir(ref_obj_a) + dir(ref_obj_b)
    elif obj_type == "Optimizer":
        ref_obj = Optimizer(1.0)
        skiplist += dir(ref_obj)
        skiplist.remove("variables")
    elif obj_type == "Loss":
        ref_obj = Loss()
        skiplist += dir(ref_obj)
    else:
        raise ValueError(
            f"get_attr_skiplist got invalid {obj_type=}. "
            "Accepted values for `obj_type` are "
            "['Layer', 'Functional', 'Sequential', 'Metric', "
            "'Optimizer', 'Loss']"
        )

    global_state.set_global_attribute(
        f"saving_attr_skiplist_{obj_type}", skiplist
    )
    return skiplist
