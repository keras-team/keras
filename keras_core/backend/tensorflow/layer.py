import tensorflow as tf


class TFLayer(tf.__internal__.tracking.AutoTrackable):
    
    @property
    def _default_save_signature(self):
        """For SavedModel support: returns the default serving signature."""

        shapes_dict = self._build_shapes_dict
        if len(shapes_dict) == 1:
            input_shape = tuple(shapes_dict.values())[0]
            input_signature = [tf.TensorSpec(input_shape, self.compute_dtype)]
        else:
            input_signature = [tf.nest.map_structure(lambda x: tf.TensorSpec(x.shape, self.compute_dtype), shapes_dict)]

        @tf.function(input_signature=input_signature)
        def serving_default(inputs):
            return self(inputs)
        return serving_default


