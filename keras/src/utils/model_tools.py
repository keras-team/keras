"""Utilities related to model inspection and patching."""

import io
import json
import base64
import zipfile
import matplotlib.pyplot as plt

from difflib import unified_diff
from IPython.display import display, HTML
from diff_match_patch import diff_match_patch

from keras.src.saving import load_model
from keras.src.api_export import keras_export
from keras.src.saving import deserialize_keras_object
from keras.src.saving.saving_lib import H5IOStore

_CONFIG_FILENAME = "config.json"
_METADATA_FILENAME = "metadata.json"
_VARS_FNAME = "model.weights"


def load_model_config(model_path):
    """ Auxiliar function.
        Loads model and extract its JSON configuration."""
    model = load_model(model_path)
    return model.to_json()


def compare_model_configs(config1_json, config2_json):
    """ Auxiliar function.
        Returns differences between two model configurations."""
    config1 = json.loads(config1_json)
    config2 = json.loads(config2_json)
    if config1["config"]["name"]:
        del config1["config"]["name"]
    if config2["config"]["name"]:
        del config2["config"]["name"]

    config1_str = json.dumps(config1, indent=2)
    config2_str = json.dumps(config2, indent=2)

    diff = list(unified_diff(config1_str.splitlines(),
                             config2_str.splitlines(),
                             fromfile='model1',
                             tofile='model2'
                             )
                )
    return diff


def diff_text(text1, text2):
    """ Auxiliar function.
        Returns highlighted differences between two texts."""
    style1 = "background-color:rgba(212, 237, 218, 0.5); color:#155724;"
    style2 = "background-color:rgba(248, 215, 218, 0.5); color:#721c24;"
    dmp = diff_match_patch()
    diffs = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemantic(diffs)
    highlighted1, highlighted2 = [], []
    for (op, data) in diffs:
        if op == 0:  # equal, no highlighting
            highlighted1.append(data)
            highlighted2.append(data)
        elif op == 1:  # highlights with green color
            highlighted2.append(f"<span style='{style1}'>{data}</span>")
        elif op == -1:  # highlights with red color
            highlighted1.append(f"<span style='{style2}'>{data}</span>")
    return ''.join(highlighted1), ''.join(highlighted2)


def render_differences(diff):
    """ Auxiliar function.
        Renders differences between two model configurations."""

    base_style = "border:1px solid black; padding: 10px; text-align: left;"

    html_output = ["<h3>Model Configuration Differences:</h3>",
                   "<table style='width:100%; border-collapse:collapse;'>"]
    html_output.append(f"<tr><th style='{base_style} width:50%;'>Model 1</th>\
                        <th style='{base_style} width:50%;'>Model 2</th></tr>")

    if diff:
        text1_lines, text2_lines = [], []
        for line in diff:
            if line.startswith('- '):
                text1_lines.append(line[2:])
            elif line.startswith('+ '):
                text2_lines.append(line[2:])
        text1 = "\n".join(text1_lines)
        text2 = "\n".join(text2_lines)
        highlighted1, highlighted2 = diff_text(text1, text2)
        html_output.append(f"<tr><td style='{base_style} vertical-align: top;'>\
                           <pre>{highlighted1}</pre></td>")
        html_output.append(f"<td style='{base_style} vertical-align: top;'> \
                           <pre>{highlighted2}</pre></td></tr>")
    else:
        html_output.append(f"<tr><td colspan='2' style='{base_style}'> \
                           No differences found</td></tr>")

    html_output.append("</table>")
    display(HTML(''.join(html_output)))


@keras_export("keras.utils.compare_models")
def compare_models(model1_path, model2_path):
    """ Compares two models.
        Shows them side by side with differences highlighted."""
    config1_json = load_model_config(model1_path)
    config2_json = load_model_config(model2_path)

    diff = compare_model_configs(config1_json, config2_json)
    render_differences(diff)


def create_color_grid(weights):
    """ Auxiliar function.
        Create a heatmap of the weights."""

    fig, ax = plt.subplots(figsize=(5, 5))
    if weights.ndim > 2:
        weights = weights.reshape(-1, weights.shape[-1])
    if weights.ndim == 1:
        cax = ax.imshow(weights.reshape(-1, 1), aspect='auto', cmap='viridis')
    else:
        cax = ax.matshow(weights, interpolation='nearest', cmap='viridis')
        if weights.shape[0] / weights.shape[1] > 10 or \
                weights.shape[1] / weights.shape[0] > 10:
            ax.set_aspect('auto', 'datalim')
        if weights.shape[0] == 1:
            ax.set_yticks([])
        if weights.shape[1] == 1:
            ax.set_xticks([])

    fig.colorbar(cax)
    ax.set_title('Weights Heatmap')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=1)
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{image_base64}" \
                style="display:block;margin:auto;" />'


@keras_export("keras.utils.inspect_file")
def inspect_file(filepath, reference_model=None, custom_objects=None):
    """ Inspects the contents of a Keras model or weights file."""

    filepath = str(filepath)
    html_output = []

    if filepath.endswith(".keras"):
        with zipfile.ZipFile(filepath, "r") as zf:
            html_output.append(f"<h2>Keras model file '{filepath}'</h2>")

            with zf.open(_CONFIG_FILENAME, "r") as f:
                config = json.loads(f.read())
                html_output.append(
                    f"<p>Model: {config['class_name']} \
                        name='{config['config']['name']}'</p>"
                )
            if reference_model is None:
                reference_model = deserialize_keras_object(
                    config, custom_objects=custom_objects
                )

            with zf.open(_METADATA_FILENAME, "r") as f:
                metadata = json.loads(f.read())
                html_output.append(f"<p>Saved with Keras \
                                   {metadata['keras_version']}</p>")
                html_output.append(f"<p>Date saved: \
                                   {metadata['date_saved']}</p>")

            archive = zipfile.ZipFile(filepath, "r")
            weights_store = H5IOStore(
                _VARS_FNAME + ".h5", archive=archive, mode="r"
            )
            html_output.append("<h3>Weights file:</h3>")
            html_output.append(inspect_nested_dict(weights_store.h5_file))

    elif filepath.endswith(".weights.h5"):
        html_output.append(f"<h2>Keras weights file '{filepath}'</h2>")
        weights_store = H5IOStore(filepath, mode="r")
        html_output.append(inspect_nested_dict(weights_store.h5_file))

    else:
        raise ValueError(
            "Invalid filename: expected a `.keras` or `.weights.h5` extension. "
            f"Received: filepath={filepath}"
        )

    display(HTML(''.join(html_output)))


def inspect_nested_dict(store):
    """ Auxiliar function.
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
                html_output.append(f"<details style='{indent_style}'>\
                    <summary>{key}</summary>")
                html_output.append(inspect_nested_dict(value))
                html_output.append("</details>")
        else:
            w = value[()]
            shape_info = f"{w.shape} {w.dtype}"
            if w.ndim == 0:
                html_output.append(f"<details style='{indent_style}'>\
                    <summary>{key}: {shape_info}</summary></details>")
            else:
                color_grid = create_color_grid(w)
                html_output.append(f"<details style='\
                    {indent_style}'>\
                        <summary>{key}: {shape_info}</summary>\
                        {color_grid}\
                    </details>")

    return ''.join(html_output)


@keras_export("keras.utils.change_layer_name")
def change_layer_name(filepath):
    """ Interactively changes the name of a layer in a Keras model."""

    try:
        model = load_model(filepath)
    except Exception as e:
        print(f"Error loading the model: {e}\n")
        return

    layers = model.layers
    print("available layers: ")
    for layer in layers:
        print(layer.name)
    found = False
    layer_name = input("Enter the layer name you want to change: ")
    for layer in layers:
        if layer.name == layer_name:
            found = True
            new_name = input("Enter the new layer name: ")
            layer.name = new_name
    if not found:
        print(f"Layer '{layer_name}' not found in the model.\n")
        return
    new_filepath = input("Enter the new file path to save the model: ")
    if not new_filepath.endswith(".keras"):
        new_filepath += ".keras"
    model.save(new_filepath)

    print(f"\nLayer name changed successfully. \
        Model saved at '{new_filepath}'\n")


@keras_export("keras.utils.patch_weight")
def patch_weight(filepath, layer_name, weight_index, new_value, save_path):
    """ Change the weight of a model in a specified layer at a given index."""

    def change_weight(weights, weight_index, new_value):
        element = weights
        for index in weight_index[:-1]:
            element = element[index]
        element[weight_index[-1]] = new_value
        return weights

    if filepath.endswith(".keras"):
        model = load_model(filepath)
        try:
            layer = model.get_layer(name=layer_name)
        except ValueError:
            print(f"Layer {layer_name} not found in the model.")
            return

        weights = layer.get_weights()
        weights = change_weight(weights, weight_index, new_value)
        layer.set_weights(weights)
        model.save(save_path)
        print(f"Model saved to {save_path}")
    else:
        raise ValueError(
            "Invalid filename: expected a `.keras` or `.weights.h5` extension."
            f"Received: filepath={filepath}"
        )
