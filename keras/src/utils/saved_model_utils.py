import json
import zipfile

import h5py

from keras.saving import deserialize_keras_object
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
            self.init_for_keras(custom_objects, filepath, reference_model)
        elif filepath.endswith(".weights.h5"):
            pass
        else:
            raise ValueError(
                "Invalid filename: "
                "expected a `.keras` `.weights.h5` extension. "
                f"Received: filepath={filepath}"
            )

        def recursive_search(data):
            result = {}
            for key in data.keys():
                value = data[key]
                if isinstance(value, h5py.Group) and len(value) == 0:
                    continue
                if hasattr(value, "keys"):
                    result[key] = recursive_search(value)
                else:
                    result[key] = value
            return result

        archive = zipfile.ZipFile(filepath, "r")
        weights_store = H5IOStore(
            _VARS_FNAME + ".h5", archive=archive, mode="r"
        )
        self.nested_dict = recursive_search(weights_store.h5_file)

    def init_for_keras(self, custom_objects, filepath, reference_model):
        with zipfile.ZipFile(filepath, "r") as zf:
            with zf.open(_CONFIG_FILENAME, "r") as f:
                self.config = json.loads(f.read())
            if reference_model is None:
                self.reference_model = deserialize_keras_object(
                    self.config, custom_objects=custom_objects
                )

            with zf.open(_METADATA_FILENAME, "r") as f:
                self.metadata = json.loads(f.read())

    def list_layer_paths(self):
        layer_key = "layers"
        layer_paths = [
            f"{layer_key}/{key}"
            for key in self.nested_dict[layer_key].keys()
            if "vars" in self.nested_dict[layer_key][key]
        ]
        return layer_paths
