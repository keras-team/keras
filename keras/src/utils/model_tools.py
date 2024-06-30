"""Utilities related to model inspection, comparison and patching."""

import base64
import io
import json
import os
import zipfile

import matplotlib.pyplot as plt
import rich
from IPython.display import HTML
from IPython.display import display

from keras.src.api_export import keras_export
from keras.src.saving import deserialize_keras_object
from keras.src.saving import load_model
from keras.src.saving.saving_lib import H5IOStore

_CONFIG_FILENAME = "config.json"
_METADATA_FILENAME = "metadata.json"
_VARS_FNAME = "model.weights"
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_BLUE = "\033[94m"
ANSI_RESET = "\033[0m"


def is_notebook():
    """Auxiliar function.
    Detects if the code is running in a Jupyter notebook."""

    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # jupyter notebook
        elif shell == "TerminalInteractiveShell":
            return False  # terminal
        else:
            return False
    except NameError:
        return False


def create_color_grid(weights):
    """Auxiliar function.
    Create a heatmap of the weights."""

    fig, ax = plt.subplots(figsize=(5, 5))
    if weights.ndim > 2:
        weights = weights.reshape(-1, weights.shape[-1])
    if weights.ndim == 1:
        cax = ax.imshow(weights.reshape(-1, 1), aspect="auto", cmap="viridis")
    else:
        cax = ax.matshow(weights, interpolation="nearest", cmap="viridis")
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
    ax.set_title("Weights Heatmap")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=1)
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{image_base64}" \
                style="display:block;margin:auto;" />'


def inspect_nested_dict_html(store):
    """Auxiliar function.
    Inspect the contents of a nested dictionary."""

    html_output = []
    indent_style = "margin-left: 20px;"

    for key in store.keys():
        value = store[key]

        if hasattr(value, "keys"):
            skip = False
            if (
                list(value.keys()) == ["vars"]
                and len(value["vars"].keys()) == 0
            ):
                skip = True
            if key == "vars" and len(value.keys()) == 0:
                skip = True
            if not skip:
                html_output.append(
                    f"<details style='{indent_style}'>\
                    <summary>{key}</summary>"
                )
                html_output.append(inspect_nested_dict_html(value))
                html_output.append("</details>")
        else:
            w = value[()]
            shape_info = f"{w.shape} {w.dtype}"
            if w.ndim == 0:
                html_output.append(
                    f"<details style='{indent_style}'>\
                    <summary>{key}: {shape_info}</summary></details>"
                )
            else:
                color_grid = create_color_grid(w)
                html_output.append(
                    f"<details style='\
                    {indent_style}'>\
                        <summary>{key}: {shape_info}</summary>\
                        {color_grid}\
                    </details>"
                )

    return "".join(html_output)


def inspect_nested_dict_shell(store, indent=0):
    """Auxiliar function.
    Inspect the contents of a nested dictionary."""

    output = []
    indent_str = " " * (indent * 4)

    for key in store.keys():
        value = store[key]
        if hasattr(value, "keys"):
            skip = False
            if (
                list(value.keys()) == ["vars"]
                and len(value["vars"].keys()) == 0
            ):
                skip = True
            if key == "vars" and len(value.keys()) == 0:
                skip = True
            if not skip:
                output.append(f"{ANSI_BLUE}{indent_str}{key}:{ANSI_RESET}")
                output.append(inspect_nested_dict_shell(value, indent + 1))
        else:
            w = value[()]
            shape_info = f"{w.shape} {w.dtype}"
            output.append(
                f"{ANSI_GREEN}{indent_str}{key}: \
                          {shape_info}{ANSI_RESET}"
            )
    return "\n".join(output)


@keras_export("keras.utils.inspect_file")
def inspect_file(filepath, ref_model=None, cust_objs=None):
    """Inspects the contents of a Keras model or weights file."""

    filepath = str(filepath)
    inNotebook = is_notebook()
    output = []

    if filepath.endswith(".keras"):
        if inNotebook:
            output.append(f"<h2>Keras model file '{filepath}'</h2>")
        else:
            output.append(
                f"{ANSI_BLUE}Keras model file \
                          '{filepath}'{ANSI_RESET}"
            )

        with zipfile.ZipFile(filepath, "r") as zf:
            with zf.open(_CONFIG_FILENAME, "r") as f:
                config = json.loads(f.read())
                if inNotebook:
                    output.append(
                        f"<p>Model: {config['class_name']} \
                                  name='{config['config']['name']}'</p>"
                    )
                else:
                    output.append(
                        f"{ANSI_GREEN}Model: {config['class_name']} \
                    name='{config['config']['name']}'{ANSI_RESET}"
                    )
            if ref_model is None:
                ref_model = deserialize_keras_object(
                    config, custom_objects=cust_objs
                )

            with zf.open(_METADATA_FILENAME, "r") as f:
                metadata = json.loads(f.read())
                if inNotebook:
                    output.append(
                        f"<p>Saved with Keras \
                                  {metadata['keras_version']}</p>"
                    )
                    output.append(
                        f"<p>Date saved:\
                        {metadata['date_saved']}</p>"
                    )
                else:
                    output.append(
                        f"{ANSI_GREEN}Saved with Keras \
                                  {metadata['keras_version']}{ANSI_RESET}"
                    )
                    output.append(
                        f"{ANSI_GREEN}Date saved:\
                        {metadata['date_saved']}{ANSI_RESET}"
                    )

            archive = zipfile.ZipFile(filepath, "r")
            weights_store = H5IOStore(
                _VARS_FNAME + ".h5", archive=archive, mode="r"
            )
            if inNotebook:
                output.append("<h3>Weights file:</h3>")
                output.append(inspect_nested_dict_html(weights_store.h5_file))
            else:
                output.append(f"{ANSI_BLUE}Weights file:{ANSI_RESET}")
                output.append(inspect_nested_dict_shell(weights_store.h5_file))

    elif filepath.endswith(".weights.h5"):
        if inNotebook:
            output.append(f"<h2>Keras weights file '{filepath}'</h2>")
            weights_store = H5IOStore(filepath, mode="r")
            output.append(inspect_nested_dict_html(weights_store.h5_file))
        else:
            output.append(
                f"{ANSI_BLUE}Keras weights file \
                '{filepath}'{ANSI_RESET}"
            )
            weights_store = H5IOStore(filepath, mode="r")
            output.append(inspect_nested_dict_shell(weights_store.h5_file))

    else:
        raise ValueError(
            f"Invalid filename: expected a\
            `.keras` or `.weights.h5` extension.\
                Received: filepath={filepath}"
        )

    final_output = "".join(output) if inNotebook else "\n".join(output)
    if inNotebook:
        display(HTML(final_output))
    else:
        print(final_output)


def extract_relevant_info(model_path):
    """Loads model and extracts its layer names,
    number of weights, and sublayers."""
    model = load_model(model_path)
    model_info = []
    for layer in model.layers:
        config = layer.get_config()
        name = layer.name
        weights = layer.get_weights()
        weights_info = [
            {"shape": w.shape, "dtype": str(w.dtype)} for w in weights
        ]
        num_sublayers = len(config.get("layers", []))
        model_info.append(
            {
                "name": name,
                "num_weights": len(weights_info),
                "num_sublayers": num_sublayers,
                "weights": weights_info,
            }
        )
    return model_info


def compare_layers(model1_layers, model2_layers):
    """Returns matching and different layers between two models."""
    names1 = {layer["name"]: layer for layer in model1_layers}
    names2 = {layer["name"]: layer for layer in model2_layers}
    m1_layers, m2_layers = set(names1), set(names2)

    set_added = m2_layers - m1_layers  # in model 2 but not in model 1
    set_removed = m1_layers - m2_layers  # in model 1 but not in model 2
    set_match = m1_layers & m2_layers  # common

    added_layers = {name: names2[name] for name in set_added}
    removed_layers = {name: names1[name] for name in set_removed}
    matching_layers = {name: (names1[name], names2[name]) for name in set_match}

    return added_layers, removed_layers, matching_layers


def generate_layer_row(name, details1, details2):
    """Helper function to generate a row for a layer with given details."""
    return [name, details1, details2]


def generate_weight_diff_rows(info1, info2):
    """Helper function to generate weight difference rows."""
    weight_diffs = []
    for w1, w2 in zip(info1["weights"], info2["weights"]):
        if w1["shape"] != w2["shape"] or w1["dtype"] != w2["dtype"]:
            weight_diffs.append(
                generate_layer_row(
                    "\tShape:", f"{w1['shape']}", f"{w2['shape']}"
                )
            )
            weight_diffs.append(
                generate_layer_row(
                    "\tDtype:", f"{w1['dtype']}", f"{w2['dtype']}"
                )
            )
    return weight_diffs


def render_differences(m1_name, m2_name, added, removed, matching):
    """Renders differences and matching layers
    between two lists of layer information."""
    console = rich.console.Console()
    table = rich.table.Table(show_header=True, header_style="bold turquoise2")
    table.add_column("Layer Name", justify="left")
    table.add_column(f"{m1_name} details", justify="left")
    table.add_column(f"{m2_name} details", justify="left")

    if added or removed:
        table.add_row("Non matching Layers", "", "", style="bold red3")
        if added:
            for name, info in added.items():
                table.add_row(
                    name,
                    "Absent",
                    f"Weights: {info['num_weights']},\
                    Sublayers: {info['num_sublayers']}",
                )

        if removed:
            for name, info in removed.items():
                table.add_row(
                    name,
                    f"Weights: {info['num_weights']},\
                    Sublayers: {info['num_sublayers']}",
                    "Absent",
                )
        table.add_row("", "", "")

    if matching:
        table.add_row("Matching Layers", "", "", style="bold chartreuse1")
        for name, (info1, info2) in matching.items():
            table.add_row("", "", "")
            if (
                info1["weights"] == info2["weights"]
                and info1["num_sublayers"] == info2["num_sublayers"]
            ):
                table.add_row(
                    name,
                    f"Identical layer: \
                    Weights: {info1['num_weights']}, \
                        Sublayers: {info1['num_sublayers']}",
                    "",
                )
            else:
                table.add_row(
                    name,
                    f"Weights: {info1['num_weights']}, \
                    Sublayers: {info1['num_sublayers']}",
                    f"Weights:\
                        {info2['num_weights']}, \
                            Sublayers: {info2['num_sublayers']}",
                )
                weight_diff_rows = generate_weight_diff_rows(info1, info2)
                for row in weight_diff_rows:
                    table.add_row(*row)

    console.print(table)


@keras_export("keras.utils.compare_models")
def compare_models(model1_path, model2_path):
    """Compares two models. Shows them side by side with differences
    highlighted."""
    model1_name = os.path.basename(model1_path).replace(".keras", "")
    model2_name = os.path.basename(model2_path).replace(".keras", "")

    model1_info = extract_relevant_info(model1_path)
    model2_info = extract_relevant_info(model2_path)

    added, removed, matching = compare_layers(model1_info, model2_info)
    render_differences(model1_name, model2_name, added, removed, matching)


@keras_export(["keras.KerasFileEditor", "keras.utils.KerasFileEditor"])
class KerasFileEditor:
    def __init__(self, filepath):
        self.filepath = filepath
        try:
            self.model = load_model(filepath)
        except Exception as e:
            print(f"Error loading the model: {e}")
            self.model = None

    def list_layer_paths(self):
        """List all layer names in the model."""
        if not self.model:
            return
        layers = self.model.layers
        for layer in layers:
            print(layer.name)

    def layer_info(self, layer_name):
        """Prints the weight structure for the specified layer."""
        if not self.model:
            return
        try:
            layer = self.model.get_layer(name=layer_name)
            for weight in layer.weights:
                print(weight.name, weight.shape)
        except ValueError:
            print(f"Layer '{layer_name}' not found in the model.")

    def edit_layer(self, layer_name, new_name=None, new_vars=None):
        """Edit the layer name and optionally update weights."""
        if not self.model:
            return
        try:
            layer = self.model.get_layer(name=layer_name)
            if new_name:
                layer.name = new_name  # Internal API to change the layer name
            if new_vars:
                if len(layer.weights) != len(new_vars):
                    print(
                        f"Number of new variables ({len(new_vars)}) does \
                        not match the number of weights in the layer \
                            ({len(layer.weights)})."
                    )
                    return
                for weight, new_weight in zip(layer.weights, new_vars):
                    if weight.shape != new_weight.shape:
                        print(
                            f"Shape mismatch: weight shape {weight.shape}, \
                            new weight shape {new_weight.shape}."
                        )
                        return
                    weight.assign(new_weight)
        except ValueError:
            print(f"Layer '{layer_name}' not found in the model.")

    def write_out(self, new_filepath):
        """Save the model to a new file."""
        if not self.model:
            return
        if not new_filepath.endswith(".keras"):
            new_filepath += ".keras"
        self.model.save(new_filepath)
        print(f"Model saved at '{new_filepath}'.")
