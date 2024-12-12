import tensorflow as tf

from keras.src import layers


class TFExportArchive:
    def track(self, resource):
        if not isinstance(resource, tf.__internal__.tracking.Trackable):
            raise ValueError(
                "Invalid resource type. Expected an instance of a "
                "TensorFlow `Trackable` (such as a Keras `Layer` or `Model`). "
                f"Received instead an object of type '{type(resource)}'. "
                f"Object received: {resource}"
            )

        if isinstance(resource, layers.Layer):
            # Variables in the lists below are actually part of the trackables
            # that get saved, because the lists are created in __init__.
            variables = resource.variables
            trainable_variables = resource.trainable_variables
            non_trainable_variables = resource.non_trainable_variables
            self._tf_trackable.variables += variables
            self._tf_trackable.trainable_variables += trainable_variables
            self._tf_trackable.non_trainable_variables += (
                non_trainable_variables
            )

    def add_endpoint(self, name, fn, input_signature=None, **kwargs):
        decorated_fn = tf.function(
            fn, input_signature=input_signature, autograph=False
        )
        return decorated_fn
