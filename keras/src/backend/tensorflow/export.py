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
        if isinstance(input_signature, dict):
            # Create a wrapper that handles both dict and positional args.
            def wrapped_fn(*args, **kwargs):
                if len(args) == 1 and isinstance(args[0], dict):
                    # Handle case where input is passed as a single dict.
                    return fn(args[0])
                else:
                    # Handle case where inputs are passed as positional args
                    # but need to be converted to dict for the model.
                    dict_inputs = {
                        name: arg 
                        for name, arg in zip(input_signature.keys(), args)
                    }
                    return fn(dict_inputs)
            # Convert dict input_signature to list of TensorSpecs while preserving order.
            flat_input_signature = [
                input_signature[name] for name in input_signature.keys()
            ]
            decorated_fn = tf.function(
                wrapped_fn, 
                input_signature=flat_input_signature,
                autograph=False
            )
        else:
            decorated_fn = tf.function(
                fn, 
                input_signature=input_signature,
                autograph=False
            )
        return decorated_fn
