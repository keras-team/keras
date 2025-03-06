from keras.src import layers
from keras.src import tree


class MlxExportArchive:
    def __init__(self):
        self._backend_variables = []
        self._backend_trainable_variables = []
        self._backend_non_trainable_variables = []

    def track(self, resource):
        if not isinstance(resource, layers.Layer):
            raise ValueError(
                "Invalid resource type. Expected an instance of a "
                "MLX-based Keras `Layer` or `Model`. "
                f"Received instead an object of type '{type(resource)}'. "
                f"Object received: {resource}"
            )

        if isinstance(resource, layers.Layer):
            # Variables in the lists below are actually part of the trackables
            # that get saved, because the lists are created in __init__.
            trainable_variables = resource.trainable_variables
            non_trainable_variables = resource.non_trainable_variables

            self._tf_trackable.trainable_variables += tree.map_structure(
                self._convert_to_tf_variable, trainable_variables
            )
            self._tf_trackable.non_trainable_variables += tree.map_structure(
                self._convert_to_tf_variable, non_trainable_variables
            )
            self._tf_trackable.variables = (
                self._tf_trackable.trainable_variables
                + self._tf_trackable.non_trainable_variables
            )

            self._backend_trainable_variables += trainable_variables
            self._backend_non_trainable_variables += non_trainable_variables
            self._backend_variables = (
                self._backend_trainable_variables
                + self._backend_non_trainable_variables
            )

    def add_endpoint(self, name, fn, input_signature=None, **kwargs):
        raise NotImplementedError(
            "`add_endpoint` is not implemented in the mlx backend."
        )
