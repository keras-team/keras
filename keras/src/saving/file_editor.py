import collections
import json
import pprint
import zipfile

import h5py
import rich.console

from keras.src.saving import saving_lib
from keras.src.saving.saving_lib import H5IOStore
from keras.src.utils import summary_utils

try:
    import IPython as ipython
except ImportError:
    ipython = None


def is_ipython_notebook():
    """Checks if the code is being executed in a notebook."""
    try:
        from IPython import get_ipython

        # Check if an active IPython shell exists.
        if get_ipython() is not None:
            return True
        return False
    except ImportError:
        return False


class KerasFileEditor:
    def __init__(
        self,
        filepath,
    ):
        self.filepath = filepath
        self.metadata = None
        self.config = None
        self.model = None
        self.console = rich.console.Console(highlight=False)

        if filepath.endswith(".keras"):
            zf = zipfile.ZipFile(filepath, "r")
            weights_store = H5IOStore(
                saving_lib._VARS_FNAME + ".h5",
                archive=zf,
                mode="r",
            )
            with zf.open(saving_lib._CONFIG_FILENAME, "r") as f:
                config_json = f.read()
            with zf.open(saving_lib._METADATA_FILENAME, "r") as f:
                metadata_json = f.read()
            self.config = json.loads(config_json)
            self.metadata = json.loads(metadata_json)

        elif filepath.endswith(".weights.h5"):
            weights_store = H5IOStore(filepath, mode="r")
        else:
            raise ValueError(
                "Invalid filename: "
                "expected a `.keras` `.weights.h5` extension. "
                f"Received: filepath={filepath}"
            )

        weights_dict, object_metadata = self._extract_weights_from_store(
            weights_store.h5_file
        )
        self.weights_dict = weights_dict
        self.object_metadata = object_metadata  # {path: object_name}
        self.console.print(self._generate_filepath_info(rich_style=True))

        if self.metadata is not None:
            self.console.print(self._generate_metadata_info(rich_style=True))

    def weights_summary(self):
        if is_ipython_notebook():
            self._weights_summary_iteractive()
        else:
            self._weights_summary_cli()

    def compare_to_reference(self, model):
        # TODO
        raise NotImplementedError()

    def _edit_object(self, edit_fn, source_name, target_name=None):
        if target_name is not None and "/" in target_name:
            raise ValueError(
                "Argument `target_name` should be a leaf name, "
                "not a full path name. "
                f"Received: target_name='{target_name}'"
            )
        if "/" in source_name:
            # It's a path
            elements = source_name.split("/")
            weights_dict = self.weights_dict
            for e in elements[:-1]:
                if e not in weights_dict:
                    raise ValueError(
                        f"Path '{source_name}' not found in model."
                    )
                weights_dict = weights_dict[e]
            if elements[-1] not in weights_dict:
                raise ValueError(f"Path '{source_name}' not found in model.")
            edit_fn(
                weights_dict, source_name=elements[-1], target_name=target_name
            )
        else:
            # Ensure unicity
            def count_occurences(d, name, count=0):
                for k in d:
                    if isinstance(d[k], dict):
                        count += count_occurences(d[k], name, count)
                if name in d:
                    count += 1
                return count

            occurences = count_occurences(self.weights_dict, source_name)
            if occurences > 1:
                raise ValueError(
                    f"Name '{source_name}' occurs more than once in the model; "
                    "try passing a complete path"
                )
            if occurences == 0:
                raise ValueError(
                    f"Source name '{source_name}' does not appear in the "
                    "model. Use `editor.weights_summary()` "
                    "to list all objects."
                )

            def _edit(d):
                for k in d:
                    if isinstance(d[k], dict):
                        _edit(d[k])
                if source_name in d:
                    edit_fn(d, source_name=source_name, target_name=target_name)

            _edit(self.weights_dict)

    def rename_object(self, source_name, target_name):
        def rename_fn(weights_dict, source_name, target_name):
            weights_dict[target_name] = weights_dict[source_name]
            weights_dict.pop(source_name)

        self._edit_object(rename_fn, source_name, target_name)

    def delete_object(self, name):
        def delete_fn(weights_dict, source_name, target_name=None):
            weights_dict.pop(source_name)

        self._edit_object(delete_fn, name)

    def add_object(self, name, weights):
        if not isinstance(weights, dict):
            raise ValueError(
                "Argument `weights` should be a dict "
                "where keys are weight names (usually '0', '1', etc.) "
                "and values are NumPy arrays. "
                f"Received: type(weights)={type(weights)}"
            )

        if "/" in name:
            # It's a path
            elements = name.split("/")
            partial_path = "/".join(elements[:-1])
            weights_dict = self.weights_dict
            for e in elements[:-1]:
                if e not in weights_dict:
                    raise ValueError(
                        f"Path '{partial_path}' not found in model."
                    )
                weights_dict = weights_dict[e]
            weights_dict[elements[-1]] = weights
        else:
            self.weights_dict[name] = weights

    def delete_weight(self, object_name, weight_name):
        def delete_weight_fn(weights_dict, source_name, target_name=None):
            if weight_name not in weights_dict[source_name]:
                raise ValueError(
                    f"Weight {weight_name} not found "
                    f"in object {object_name}. "
                    "Weights found: "
                    f"{list(weights_dict[source_name].keys())}"
                )
            weights_dict[source_name].pop(weight_name)

        self._edit_object(delete_weight_fn, object_name)

    def add_weights(self, object_name, weights):
        if not isinstance(weights, dict):
            raise ValueError(
                "Argument `weights` should be a dict "
                "where keys are weight names (usually '0', '1', etc.) "
                "and values are NumPy arrays. "
                f"Received: type(weights)={type(weights)}"
            )

        def add_weight_fn(weights_dict, source_name, target_name=None):
            weights_dict[source_name].update(weights)

        self._edit_object(add_weight_fn, object_name)

    def resave_weights(self, filepath):
        filepath = str(filepath)
        if not filepath.endswith(".weights.h5"):
            raise ValueError(
                "Invalid `filepath` argument: "
                "expected a `.weights.h5` extension. "
                f"Received: filepath={filepath}"
            )
        weights_store = H5IOStore(filepath, mode="w")

        def _save(weights_dict, weights_store, inner_path):
            vars_to_create = {}
            for name, value in weights_dict.items():
                if isinstance(value, dict):
                    if value:
                        _save(
                            weights_dict[name],
                            weights_store,
                            inner_path=inner_path + "/" + name,
                        )
                else:
                    # e.g. name="0", value=HDF5Dataset
                    vars_to_create[name] = value
            if vars_to_create:
                var_store = weights_store.make(inner_path)
                for name, value in vars_to_create.items():
                    var_store[name] = value

        _save(self.weights_dict, weights_store, inner_path="")
        weights_store.close()

    def _extract_weights_from_store(self, data, metadata=None, inner_path=""):
        metadata = metadata or {}

        object_metadata = {}
        for k, v in data.attrs.items():
            object_metadata[k] = v
        if object_metadata:
            metadata[inner_path] = object_metadata

        result = collections.OrderedDict()
        for key in data.keys():
            inner_path = inner_path + "/" + key
            value = data[key]
            if isinstance(value, h5py.Group):
                if len(value) == 0:
                    continue
                if "vars" in value.keys() and len(value["vars"]) == 0:
                    continue

            if hasattr(value, "keys"):
                if "vars" in value.keys():
                    result[key], metadata = self._extract_weights_from_store(
                        value["vars"], metadata=metadata, inner_path=inner_path
                    )
                else:
                    result[key], metadata = self._extract_weights_from_store(
                        value, metadata=metadata, inner_path=inner_path
                    )
            else:
                result[key] = value
        return result, metadata

    def _generate_filepath_info(self, rich_style=False):
        if rich_style:
            filepath = f"'{self.filepath}'"
            filepath = f"{summary_utils.highlight_symbol(filepath)}"
        else:
            filepath = f"'{self.filepath}'"
        return f"Keras model file {filepath}"

    def _generate_config_info(self, rich_style=False):
        return pprint.pformat(self.config)

    def _generate_metadata_info(self, rich_style=False):
        version = self.metadata["keras_version"]
        date = self.metadata["date_saved"]
        if rich_style:
            version = f"{summary_utils.highlight_symbol(version)}"
            date = f"{summary_utils.highlight_symbol(date)}"
        return f"Saved with Keras {version} " f"- date: {date}"

    def _print_weights_structure(
        self, weights_dict, indent=0, is_first=True, prefix="", inner_path=""
    ):
        for idx, (key, value) in enumerate(weights_dict.items()):
            inner_path = inner_path + "/" + key
            is_last = idx == len(weights_dict) - 1
            if is_first:
                is_first = False
                connector = "> "
            elif is_last:
                connector = "└─ "
            else:
                connector = "├─ "

            if isinstance(value, dict):
                bold_key = summary_utils.bold_text(key)
                object_label = f"{prefix}{connector}{bold_key}"
                if inner_path in self.object_metadata:
                    metadata = self.object_metadata[inner_path]
                    if "name" in metadata:
                        name = metadata["name"]
                        object_label += f" ('{name}')"
                self.console.print(object_label)
                if is_last:
                    appended = "    "
                else:
                    appended = "│   "
                new_prefix = prefix + appended
                self._print_weights_structure(
                    value,
                    indent + 1,
                    is_first=is_first,
                    prefix=new_prefix,
                    inner_path=inner_path,
                )
            else:
                if isinstance(value, h5py.Dataset):
                    bold_key = summary_utils.bold_text(key)
                    self.console.print(
                        f"{prefix}{connector}{bold_key}:"
                        + f" shape={value.shape}, dtype={value.dtype}"
                    )
                else:
                    self.console.print(f"{prefix}{connector}{key}: {value}")

    def _weights_summary_cli(self):
        self.console.print("Weights structure")
        self._print_weights_structure(self.weights_dict, prefix=" " * 2)

    def _weights_summary_iteractive(self):

        def _generate_html_weights(dictionary, margin_left=0, font_size=20):
            html = ""
            for key, value in dictionary.items():
                if isinstance(value, dict) and value:
                    html += (
                        f'<details style="margin-left: {margin_left}px;">'
                        + f'<summary style="font-size: {font_size}px;">'
                        + f"{key}</summary>"
                        + _generate_html_weights(
                            value, margin_left + 20, font_size - 1
                        )
                        + "</details>"
                    )
                else:
                    if isinstance(value, h5py.Dataset):
                        html += (
                            f'<details style="margin-left: {margin_left}px;">'
                            + f'<summary style="font-size: {font_size}px;">'
                            + f"{key} : shape={value.shape}"
                            + f", dtype={value.dtype}</summary>"
                            + "</details>"
                        )
                    else:
                        html += (
                            f'<details style="margin-left: {margin_left}px;">'
                            + f'<summary style="font-size: {font_size}px;">'
                            + f"{key} </summary>"
                            + "</details>"
                        )
            return html

        output = "Weights structure"
        output += _generate_html_weights(self.weights_dict)

        if is_ipython_notebook():
            ipython.display.display(ipython.display.HTML(output))
