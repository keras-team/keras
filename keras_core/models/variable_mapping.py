from keras_core.layers.layer import Layer
from keras_core.metrics.metric import Metric
from keras_core.optimizers.optimizer import Optimizer
from keras_core.saving import saving_lib
from keras_core.utils import file_utils


def map_trackable_variables(trackable, store, inner_path, visited_trackables):
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
        store[inner_path + "/" + v.name] = v

    # Recursively save state of children trackables (layers, optimizers, etc.)
    for child_attr, child_obj in saving_lib._walk_trackable(trackable):
        if saving_lib._is_keras_trackable(child_obj):
            map_trackable_variables(
                child_obj,
                store,
                inner_path=file_utils.join(inner_path, child_obj.name),
                visited_trackables=visited_trackables,
            )
        elif isinstance(child_obj, (list, dict, tuple, set)):
            map_container_variables(
                child_obj,
                store,
                inner_path=file_utils.join(inner_path, child_attr),
                visited_trackables=visited_trackables,
            )


def map_container_variables(container, store, inner_path, visited_trackables):
    if isinstance(container, dict):
        container = list(container.values())

    for trackable in container:
        if saving_lib._is_keras_trackable(trackable):
            name = trackable.name
            map_trackable_variables(
                trackable,
                store,
                inner_path=file_utils.join(inner_path, name),
                visited_trackables=visited_trackables,
            )
