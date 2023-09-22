from keras.layers.layer import Layer
from keras.metrics.metric import Metric
from keras.optimizers.optimizer import Optimizer
from keras.saving import saving_lib


def map_trackable_variables(trackable, store, visited_trackables):
    # If the trackable has already been saved, skip it.
    if id(trackable) in visited_trackables:
        return

    visited_trackables.add(id(trackable))

    variables = []
    if isinstance(trackable, Layer):
        variables = (
            trackable._trainable_variables + trackable._non_trainable_variables
        )
    elif isinstance(trackable, Optimizer):
        variables = trackable._variables
    elif isinstance(trackable, Metric):
        variables = trackable._variables
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

    # Recursively save state of children trackables (layers, optimizers, etc.)
    for child_attr, child_obj in saving_lib._walk_trackable(trackable):
        if saving_lib._is_keras_trackable(child_obj):
            map_trackable_variables(
                child_obj,
                store,
                visited_trackables=visited_trackables,
            )
        elif isinstance(child_obj, (list, dict, tuple, set)):
            map_container_variables(
                child_obj,
                store,
                visited_trackables=visited_trackables,
            )


def map_container_variables(container, store, visited_trackables):
    if isinstance(container, dict):
        container = list(container.values())

    for trackable in container:
        if saving_lib._is_keras_trackable(trackable):
            map_trackable_variables(
                trackable,
                store,
                visited_trackables=visited_trackables,
            )
