import warnings

import numpy as np


def load_weights_from_hdf5_group(f, model):
    """Implements topological (order-based) weight loading.

    Args:
        f: A pointer to a HDF5 group.
        model: Model instance.

    Raises:
        ValueError: in case of mismatch between provided layers
            and weights file.
    """
    if "keras_version" in f.attrs:
        original_keras_version = f.attrs["keras_version"]
        if hasattr(original_keras_version, "decode"):
            original_keras_version = original_keras_version.decode("utf8")
    else:
        original_keras_version = "1"
    if "backend" in f.attrs:
        original_backend = f.attrs["backend"]
        if hasattr(original_backend, "decode"):
            original_backend = original_backend.decode("utf8")
    else:
        original_backend = None

    filtered_layers = []
    for layer in model.layers:
        weights = _legacy_weights(layer)
        if weights:
            filtered_layers.append(layer)

    layer_names = load_attributes_from_hdf5_group(f, "layer_names")
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, "weight_names")
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names
    if len(layer_names) != len(filtered_layers):
        raise ValueError(
            "Layer count mismatch when loading weights from file. "
            f"Model expected {len(filtered_layers)} layers, found "
            f"{len(layer_names)} saved layers."
        )

    for k, name in enumerate(layer_names):
        g = f[name]
        layer = filtered_layers[k]
        symbolic_weights = _legacy_weights(layer)
        weight_values = load_subset_weights_from_hdf5_group(g)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError(
                f"Weight count mismatch for layer #{k} (named {layer.name} in "
                f"the current model, {name} in the save file). "
                f"Layer expects {len(symbolic_weights)} weight(s). Received "
                f"{len(weight_values)} saved weight(s)"
            )
        for ref_v, val in zip(symbolic_weights, weight_values):
            ref_v.assign(val)

    if "top_level_model_weights" in f:
        symbolic_weights = list(
            v
            for v in model._trainable_variables + model._non_trainable_variables
            if v in model.weights
        )
        weight_values = load_subset_weights_from_hdf5_group(
            f["top_level_model_weights"]
        )
        if len(weight_values) != len(symbolic_weights):
            raise ValueError(
                "Weight count mismatch for top-level weights when loading "
                "weights from file. "
                f"Model expects {len(symbolic_weights)} top-level weight(s). "
                f"Received {len(weight_values)} saved top-level weight(s)"
            )
        for ref_v, val in zip(symbolic_weights, weight_values):
            ref_v.assign(val)


def load_weights_from_hdf5_group_by_name(f, model, skip_mismatch=False):
    """Implements name-based weight loading (instead of topological loading).

    Layers that have no matching name are skipped.

    Args:
        f: A pointer to a HDF5 group.
        model: Model instance.
        skip_mismatch: Boolean, whether to skip loading of layers
            where there is a mismatch in the number of weights,
            or a mismatch in the shape of the weights.

    Raises:
        ValueError: in case of mismatch between provided layers
            and weights file and skip_match=False.
    """
    if "keras_version" in f.attrs:
        original_keras_version = f.attrs["keras_version"]
        if hasattr(original_keras_version, "decode"):
            original_keras_version = original_keras_version.decode("utf8")
    else:
        original_keras_version = "1"
    if "backend" in f.attrs:
        original_backend = f.attrs["backend"]
        if hasattr(original_backend, "decode"):
            original_backend = original_backend.decode("utf8")
    else:
        original_backend = None

    # New file format.
    layer_names = load_attributes_from_hdf5_group(f, "layer_names")

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in model.layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    for k, name in enumerate(layer_names):
        g = f[name]
        weight_values = load_subset_weights_from_hdf5_group(g)
        for layer in index.get(name, []):
            symbolic_weights = _legacy_weights(layer)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    warnings.warn(
                        f"Skipping loading of weights for layer #{k} (named "
                        f"{layer.name}) due to mismatch in number of weights. "
                        f"Layer expects {len(symbolic_weights)} weight(s). "
                        f"Received {len(weight_values)} saved weight(s)",
                        stacklevel=2,
                    )
                    continue
                raise ValueError(
                    f"Weight count mismatch for layer #{k} "
                    f"(named {layer.name}). "
                    f"Layer expects {len(symbolic_weights)} weight(s). "
                    f"Received {len(weight_values)} saved weight(s)"
                )
            # Set values.
            for i in range(len(weight_values)):
                expected_shape = symbolic_weights[i].shape
                received_shape = weight_values[i].shape
                if expected_shape != received_shape:
                    if skip_mismatch:
                        warnings.warn(
                            f"Skipping loading weights for layer #{k} (named "
                            f"{layer.name}) due to mismatch in shape for "
                            f"weight {symbolic_weights[i].name}. "
                            f"Weight expects shape {expected_shape}. "
                            "Received saved weight "
                            f"with shape {received_shape}",
                            stacklevel=2,
                        )
                        continue
                    raise ValueError(
                        f"Shape mismatch in layer #{k} (named {layer.name}) "
                        f"for weight {symbolic_weights[i].name}. "
                        f"Weight expects shape {expected_shape}. "
                        "Received saved weight "
                        f"with shape {received_shape}"
                    )
                else:
                    symbolic_weights[i].assign(weight_values[i])

    if "top_level_model_weights" in f:
        symbolic_weights = (
            model._trainable_weights + model._non_trainable_weights
        )
        weight_values = load_subset_weights_from_hdf5_group(
            f["top_level_model_weights"]
        )

        if len(weight_values) != len(symbolic_weights):
            if skip_mismatch:
                warnings.warn(
                    "Skipping loading top-level weights for model due to "
                    "mismatch in number of weights. "
                    f"Model expects {len(symbolic_weights)} "
                    "top-level weight(s). "
                    f"Received {len(weight_values)} saved top-level weight(s)",
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    "Weight count mismatch for top-level weights of model. "
                    f"Model expects {len(symbolic_weights)} "
                    "top-level weight(s). "
                    f"Received {len(weight_values)} saved top-level weight(s)"
                )
        else:
            for i in range(len(weight_values)):
                expected_shape = symbolic_weights[i].shape
                received_shape = weight_values[i].shape
                if expected_shape != received_shape:
                    if skip_mismatch:
                        warnings.warn(
                            "Skipping loading top-level weight for model due "
                            "to mismatch in shape for "
                            f"weight {symbolic_weights[i].name}. "
                            f"Weight expects shape {expected_shape}. "
                            "Received saved weight "
                            f"with shape {received_shape}",
                            stacklevel=2,
                        )
                    else:
                        raise ValueError(
                            "Shape mismatch in model for top-level weight "
                            f"{symbolic_weights[i].name}. "
                            f"Weight expects shape {expected_shape}. "
                            "Received saved weight "
                            f"with shape {received_shape}"
                        )
                else:
                    symbolic_weights[i].assign(weight_values[i])


def load_subset_weights_from_hdf5_group(f):
    """Load layer weights of a model from hdf5.

    Args:
        f: A pointer to a HDF5 group.

    Returns:
        List of NumPy arrays of the weight values.

    Raises:
        ValueError: in case of mismatch between provided model
            and weights file.
    """
    weight_names = load_attributes_from_hdf5_group(f, "weight_names")
    return [np.asarray(f[weight_name]) for weight_name in weight_names]


def load_attributes_from_hdf5_group(group, name):
    """Loads attributes of the specified name from the HDF5 group.

    This method deals with an inherent problem
    of HDF5 file which is not able to store
    data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

    Args:
        group: A pointer to a HDF5 group.
        name: A name of the attributes to load.

    Returns:
        data: Attributes data.
    """
    if name in group.attrs:
        data = [
            n.decode("utf8") if hasattr(n, "decode") else n
            for n in group.attrs[name]
        ]
    else:
        data = []
        chunk_id = 0
        while f"{name}{chunk_id}" in group.attrs:
            data.extend(
                [
                    n.decode("utf8") if hasattr(n, "decode") else n
                    for n in group.attrs[f"{name}{chunk_id}"]
                ]
            )
            chunk_id += 1
    return data


def _legacy_weights(layer):
    """Legacy weight order converter.

    For legacy reason, the layer.weights was in the order of
    [self.trainable_weights + self.non_trainable_weights], and this order was
    used for preserving the weights in h5 format. The new order of layer.weights
    are the same as layer.get_weights() which is more intuitive for user. To
    keep supporting the existing saved h5 file, this method should be used to
    save/load weights.

    Args:
        layer: a `Model` or `Layer` instance.

    Returns:
        A list of variables with the legacy weight order.
    """
    return layer.trainable_weights + layer.non_trainable_weights
