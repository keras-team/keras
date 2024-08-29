from keras.src.layers.layer import Layer
from keras.src.metrics.metric import Metric
from keras.src.optimizers.optimizer import Optimizer
from keras.src.saving import saving_lib
from keras.src.saving.keras_savable import KerasSavable


def map_savable_variables(savable, store, visited_savables):
    # If the savable has already been seen, skip it.
    if id(savable) in visited_savables:
        return

    visited_savables.add(id(savable))

    variables = []
    if isinstance(savable, Layer):
        variables = (
            savable._trainable_variables + savable._non_trainable_variables
        )
    elif isinstance(savable, Optimizer):
        variables = savable._variables
    elif isinstance(savable, Metric):
        variables = savable._variables
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

    # Recursively save state of children savables (layers, optimizers, etc.)
    for child_attr, child_obj in saving_lib._walk_savable(savable):
        if isinstance(child_obj, KerasSavable):
            map_savable_variables(
                child_obj,
                store,
                visited_savables=visited_savables,
            )
        elif isinstance(child_obj, (list, dict, tuple, set)):
            map_container_variables(
                child_obj,
                store,
                visited_savables=visited_savables,
            )


def map_container_variables(container, store, visited_savables):
    if isinstance(container, dict):
        container = list(container.values())

    for savable in container:
        if isinstance(savable, KerasSavable):
            map_savable_variables(
                savable,
                store,
                visited_savables=visited_savables,
            )
