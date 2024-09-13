import base64
import io
import json
import zipfile

import h5py
from IPython.core.display import HTML
from matplotlib import pyplot as plt

from keras.src.saving import deserialize_keras_object
from keras.src.saving.saving_lib import H5IOStore

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

    def list_layers(self):
        margin_left = 20
        font_size = 20
        print_prefix = (
            f'<details style="margin-left: {margin_left}px;"><summary style="font-size: {font_size}px;">'
            + "{key}</summary>"
        )
        print_suffix = "</details>"

        def _create_color_grid(weights):
            fig, ax = plt.subplots(figsize=(5, 5))
            if weights.ndim > 2:
                weights = weights.reshape(-1, weights.shape[-1])

            if weights.ndim == 1:
                cax = ax.imshow(
                    weights.reshape(1, -1), aspect="auto", cmap="viridis"
                )
            else:
                cax = ax.matshow(
                    weights, interpolation="nearest", cmap="viridis"
                )
                if (
                    weights.shape[0] / weights.shape[1] > 10
                    or weights.shape[1] / weights.shape[0] > 10
                ):
                    ax.set_aspect("auto", "datalim")
                if weights.shape[0] == 1:
                    ax.set_yticks([])
                if weights.shape[1] == 1:
                    ax.set_xticks([])

            fig.colorbar(cax)
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
            ax.set_xticks([])
            ax.set_yticks([])

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=1)
            plt.close(fig)
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            return f'<img src="data:image/png;base64,{image_base64}" style="display:block;margin:auto;" />'

        def _generate_html_config(config):
            html_output = [
                f"<p>Model: {config['class_name']} name='{config['config']['name']}'</p>"
            ]
            return "".join(html_output)

        def _generate_html_metadata(metadata):
            html_output = [
                f"<p>Saved with Keras {metadata['keras_version']}</p>",
                f"<p>Date saved: {metadata['date_saved']}</p>",
            ]
            return "".join(html_output)

        def _generate_html_weight(weight_dict):
            html_output = []
            for key, value in weight_dict.items():
                html_output.append(print_prefix.format(key=key))
                if isinstance(value, dict):
                    if "vars" in value.keys():
                        for weights_key, weights in value["vars"].items():
                            html_output.append(
                                print_prefix.format(
                                    key=f"{weights_key} : shape={weights.shape}, dtype={weights.dtype}"
                                )
                            )

                            html_output.append(_create_color_grid(weights[()]))

                            html_output.append(print_suffix)
                    else:
                        html_output.append(_generate_html_weight(value))
                html_output.append(print_suffix)

            return "".join(html_output)

        output = f"<h2>Keras model file '{self.filepath}'</h2>"

        if self.config is not None:
            config_output = _generate_html_config(self.config)
            output += config_output

        if self.metadata is not None:
            metadata_output = _generate_html_metadata(self.metadata)
            output += metadata_output

        layer_output = "<h3>Weights file:</h3>"
        layer_output += _generate_html_weight(self.nested_dict["layers"])
        output += layer_output

        display(HTML(output))
