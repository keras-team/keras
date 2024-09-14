import json
import zipfile

import h5py

from keras.src.saving import deserialize_keras_object
from keras.src.saving.saving_lib import H5IOStore

try:
    import IPython as ipython
except ImportError:
    ipython = None


def check_ipython():
    return ipython is not None


_CONFIG_FILENAME = "config.json"
_METADATA_FILENAME = "metadata.json"
_VARS_FNAME = "model.weights"


class KerasFileEditor:
    def __init__(self, filepath, reference_model=None, custom_objects=None):
        self.filepath = filepath
        self.custom_objects = custom_objects
        self.metadata = None
        self.reference_model = None
        self.config = None

        if filepath.endswith(".keras"):
            self._init_for_keras_format(
                custom_objects, filepath, reference_model
            )
            weights_store = H5IOStore(
                _VARS_FNAME + ".h5",
                archive=zipfile.ZipFile(filepath, "r"),
                mode="r",
            )
        elif filepath.endswith(".weights.h5"):
            weights_store = H5IOStore(filepath, mode="r")
        else:
            raise ValueError(
                "Invalid filename: "
                "expected a `.keras` `.weights.h5` extension. "
                f"Received: filepath={filepath}"
            )

        def _extract_values(data):
            result = {}
            for key in data.keys():
                value = data[key]
                if isinstance(value, h5py.Group) and len(value) == 0:
                    continue
                if hasattr(value, "keys"):
                    if "vars" in value.keys():
                        result[key] = _extract_values(value["vars"])
                    else:
                        result[key] = _extract_values(value)
                else:
                    result[key] = value
            return result

        self.nested_dict = _extract_values(weights_store.h5_file)

    def _init_for_keras_format(self, custom_objects, filepath, reference_model):
        with zipfile.ZipFile(filepath, "r") as zf:
            with zf.open(_CONFIG_FILENAME, "r") as f:
                self.config = json.loads(f.read())
            if reference_model is None:
                self.reference_model = deserialize_keras_object(
                    self.config, custom_objects=custom_objects
                )

            with zf.open(_METADATA_FILENAME, "r") as f:
                self.metadata = json.loads(f.read())

    def _generate_filepath_info(self):
        output = f"Keras model file '{self.filepath}'"
        return output

    def _generate_config_info(self):
        output = (
            f"Model: {self.config['class_name']} "
            + f"'name='{self.config['config']['name']}'"
        )
        return output

    def _generate_metadata_info(self):
        output = [
            f"Saved with Keras {self.metadata['keras_version']}",
            f"Date saved: {self.metadata['date_saved']}",
        ]
        return output

    def list_layers_for_cli(self):
        def _print_filepath():
            print(self._generate_filepath_info())

        def _print_config():
            print(self._generate_config_info())

        def _print_metadata():
            for meta_info in self._generate_metadata_info():
                print(meta_info)

        def _print_layer_structure(
            dictionary, indent=0, is_last=True, prefix=""
        ):
            for idx, (key, value) in enumerate(dictionary.items()):
                is_last_item = idx == len(dictionary) - 1
                connector = "└─ " if is_last_item else "├─ "

                if isinstance(value, dict):
                    print(f"{prefix}{connector}{key}")
                    new_prefix = prefix + ("    " if is_last_item else "│   ")
                    _print_layer_structure(
                        value,
                        indent + 1,
                        is_last=is_last_item,
                        prefix=new_prefix,
                    )
                else:
                    if isinstance(value, h5py.Dataset):
                        print(
                            f"{prefix}{connector}{key}:"
                            + f" shape={value.shape}, dtype={value.dtype}"
                        )
                    else:
                        print(f"{prefix}{connector}{key}: {value}")

        def _print_layer():
            print("Layers")
            _print_layer_structure(self.nested_dict["layers"], prefix=" " * 2)

        _print_filepath()

        if self.config is not None:
            _print_config()

        if self.metadata is not None:
            _print_metadata()

        _print_layer()

    def list_layers_for_html(self):
        if not check_ipython():
            message = (
                "You must install ipython (`pip install ipython`) for"
                + "KerasFileEditor to work."
            )
            raise ImportError(message)

        def _generate_html_filepath():
            output = f"<h2>{self._generate_filepath_info()}</h2>"
            return output

        def _generate_html_config():
            output = f"<p>{self._generate_config_info()}</p>"
            return output

        def _generate_html_metadata():
            output = [
                f"<p>{meta_info}</p>"
                for meta_info in self._generate_metadata_info()
            ]
            output = "".join(output)
            return output

        def _generate_html_weight(dictionary, margin_left=0, font_size=20):
            html = ""
            for key, value in dictionary.items():
                if isinstance(value, dict) and value:
                    html += (
                        f'<details style="margin-left: {margin_left}px;">'
                        + f'<summary style="font-size: {font_size}px;">'
                        + f"{key}</summary>"
                        + _generate_html_weight(
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

        def _generate_html_layer():
            output = "<p>Layers</p>"
            output += _generate_html_weight(self.nested_dict["layers"])
            return output

        output = _generate_html_filepath()

        if self.config is not None:
            output += _generate_html_config()

        if self.metadata is not None:
            output += _generate_html_metadata()

        output += _generate_html_layer()

        ipython.display.display(ipython.display.HTML(output))
