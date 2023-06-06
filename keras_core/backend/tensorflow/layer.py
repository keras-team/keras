import tensorflow as tf


class TFLayer(tf.__internal__.tracking.AutoTrackable):
    def _post_build(self):
        """Can be overriden to perform post-build actions."""
        pass

    @property
    def _default_save_signature(self):
        """For SavedModel support: returns the default serving signature."""

        from keras_core.models.functional import Functional
        from keras_core.models.model import Model
        from keras_core.models.sequential import Sequential

        if not isinstance(self, Model):
            return None

        inputs = None
        if (
            isinstance(self, Sequential)
            and getattr(self, "_functional", None) is not None
        ):
            inputs = self._functional.input
        elif isinstance(self, Functional):
            inputs = self.input

        if inputs is not None:
            input_signature = [
                tf.nest.map_structure(
                    lambda x: tf.TensorSpec(x.shape, self.compute_dtype),
                    inputs,
                )
            ]
        else:
            shapes_dict = self._build_shapes_dict
            if len(shapes_dict) == 1:
                input_shape = tuple(shapes_dict.values())[0]
                input_signature = [
                    tf.TensorSpec(input_shape, self.compute_dtype)
                ]
            else:
                input_signature = [
                    tf.nest.map_structure(
                        lambda x: tf.TensorSpec(x.shape, self.compute_dtype),
                        shapes_dict,
                    )
                ]

        @tf.function(input_signature=input_signature)
        def serving_default(inputs):
            return self(inputs)

        return serving_default
