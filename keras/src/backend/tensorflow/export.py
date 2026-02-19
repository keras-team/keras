import tensorflow as tf

from keras.src.export.saved_model_export_archive import SavedModelExportArchive


class TFExportArchive(SavedModelExportArchive):
    """TensorFlow backend implementation of SavedModel export archive."""

    def _track_layer(self, layer):
        # Variables in the lists below are actually part of the trackables
        # that get saved, because the lists are created in __init__.
        variables = layer.variables
        trainable_variables = layer.trainable_variables
        non_trainable_variables = layer.non_trainable_variables
        self._tf_trackable.variables += variables
        self._tf_trackable.trainable_variables += trainable_variables
        self._tf_trackable.non_trainable_variables += non_trainable_variables

    def _add_endpoint_helper(self, name, fn, input_signature, **kwargs):
        decorated_fn = tf.function(
            fn, input_signature=input_signature, autograph=False
        )
        return decorated_fn
