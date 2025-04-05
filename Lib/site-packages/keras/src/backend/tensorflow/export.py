import tensorflow as tf


class TFExportArchive:
    def _track_layer(self, layer):
        # Variables in the lists below are actually part of the trackables
        # that get saved, because the lists are created in __init__.
        variables = layer.variables
        trainable_variables = layer.trainable_variables
        non_trainable_variables = layer.non_trainable_variables
        self._tf_trackable.variables += variables
        self._tf_trackable.trainable_variables += trainable_variables
        self._tf_trackable.non_trainable_variables += non_trainable_variables

    def add_endpoint(self, name, fn, input_signature=None, **kwargs):
        decorated_fn = tf.function(
            fn, input_signature=input_signature, autograph=False
        )
        return decorated_fn
