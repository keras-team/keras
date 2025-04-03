from keras.src.layers.layer import Layer
from keras.src.metrics.metric import Metric
from keras.src.optimizers.optimizer import Optimizer
from keras.src.saving import saving_lib
from keras.src.saving.keras_saveable import KerasSaveable


def map_saveable_variables(saveable, store, visited_saveables):
    # If the saveable has already been seen, skip it.
    if id(saveable) in visited_saveables:
        return

    visited_saveables.add(id(saveable))

    variables = []
    if isinstance(saveable, Layer):
        variables = (
            saveable._trainable_variables + saveable._non_trainable_variables
        )
    elif isinstance(saveable, Optimizer):
        variables = saveable._variables
    elif isinstance(saveable, Metric):
        variables = saveable._variables
    for v in variables:
        if v.path in store:
            raise ValueError(
                "The model contains two variables with a duplicate path: "
                f"path='{v.path}' appears at least twice. "
                f"This path is used for {v} and for {store[v.path]}. "
                "In order to get a variable map, make sure to use "
                "unique paths/names for each variable."
            )
        store[v.path] = v

    # Recursively save state of children saveables (layers, optimizers, etc.)
    for child_attr, child_obj in saving_lib._walk_saveable(saveable):
        if isinstance(child_obj, KerasSaveable):
            map_saveable_variables(
                child_obj,
                store,
                visited_saveables=visited_saveables,
            )
        elif isinstance(child_obj, (list, dict, tuple, set)):
            map_container_variables(
                child_obj,
                store,
                visited_saveables=visited_saveables,
            )


def map_container_variables(container, store, visited_saveables):
    if isinstance(container, dict):
        container = list(container.values())

    for saveable in container:
        if isinstance(saveable, KerasSaveable):
            map_saveable_variables(
                saveable,
                store,
                visited_saveables=visited_saveables,
            )
