import collections
import json
import pprint
import zipfile

import h5py

from keras.src.saving import saving_lib
from keras.src.saving.saving_lib import H5IOStore
from keras.src.utils import io_utils

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

        self.weights_dict = self._extract_weights_from_store(
            weights_store.h5_file
        )
        io_utils.print_msg(self._generate_filepath_info())

        if self.metadata is not None:
            io_utils.print_msg(self._generate_metadata_info())

    def weights_summary(self):
        if is_ipython_notebook():
            self._weights_summary_iteractive()
        else:
            self._weights_summary_cli()

    def compare_to_reference(self, model):
        # TODO
        raise NotImplementedError()

    def delete_layer(self, layer_name):
        # TODO
        raise NotImplementedError()

    def add_layer(self, layer_name, weights):
        # TODO
        raise NotImplementedError()

    def rename_layer(self, source_name, target_name):
        # TODO
        raise NotImplementedError()

    def delete_weight(self, layer_name, weight_name):
        # TODO
        raise NotImplementedError()

    def add_weight(self, layer_name, weight_name, weight_value):
        # TODO
        raise NotImplementedError()

    def resave_weights(self, fpath):
        # TODO
        raise NotImplementedError()

    def _extract_weights_from_store(self, data):
        result = collections.OrderedDict()
        for key in data.keys():
            value = data[key]
            if isinstance(value, h5py.Group):
                if len(value) == 0:
                    continue
                if "vars" in value.keys() and len(value["vars"]) == 0:
                    continue

            if hasattr(value, "keys"):
                if "vars" in value.keys():
                    result[key] = self._extract_weights_from_store(
                        value["vars"]
                    )
                else:
                    result[key] = self._extract_weights_from_store(value)
            else:
                result[key] = value
        return result

    def _generate_filepath_info(self):
        return f"Keras model file '{self.filepath}'"

    def _generate_config_info(self):
        return pprint.pformat(self.config)

    def _generate_metadata_info(self):
        return (
            f"Saved with Keras {self.metadata['keras_version']} "
            f"- date: {self.metadata['date_saved']}"
        )

    def _print_weights_structure(
        self, weights_dict, indent=0, is_last=True, prefix=""
    ):
        for idx, (key, value) in enumerate(weights_dict.items()):
            is_last_item = idx == len(weights_dict) - 1
            connector = "└─ " if is_last_item else "├─ "

            if isinstance(value, dict):
                io_utils.print_msg(f"{prefix}{connector}{key}")
                new_prefix = prefix + ("    " if is_last_item else "│   ")
                self._print_weights_structure(
                    value,
                    indent + 1,
                    is_last=is_last_item,
                    prefix=new_prefix,
                )
            else:
                if isinstance(value, h5py.Dataset):
                    io_utils.print_msg(
                        f"{prefix}{connector}{key}:"
                        + f" shape={value.shape}, dtype={value.dtype}"
                    )
                else:
                    io_utils.print_msg(f"{prefix}{connector}{key}: {value}")

    def _weights_summary_cli(self):
        io_utils.print_msg("Weights structure")
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
