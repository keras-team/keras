import tensorflow as tf

from keras.src.export.saved_model_export_archive import SavedModelExportArchive
from keras.src.export.saved_model_export_archive import (
    _list_variables_used_by_fns,
)


class TFExportArchive(SavedModelExportArchive):
    """TensorFlow backend implementation of SavedModel export archive."""

    def _backend_track_layer(self, layer):
        # Variables in the lists below are actually part of the trackables
        # that get saved, because the lists are created in __init__.
        variables = layer.variables
        trainable_variables = layer.trainable_variables
        non_trainable_variables = layer.non_trainable_variables
        self._tf_trackable.variables += variables
        self._tf_trackable.trainable_variables += trainable_variables
        self._tf_trackable.non_trainable_variables += non_trainable_variables

    def _backend_add_endpoint(self, name, fn, input_signature, **kwargs):
        decorated_fn = tf.function(
            fn, input_signature=input_signature, autograph=False
        )
        return decorated_fn

    def _filter_and_track_resources(self):
        # Under the TensorFlow backend, endpoint functions
        # capture the underlying
        # TensorFlow variables used by the Keras Variable
        # wrappers registered by
        # `track()`. These captured variables must be
        # tracked directly so that
        # TensorFlow recognizes them as tracked resources during
        # SavedModel export.
        #
        # Replacing the variable collections with the
        # endpoint-captured variables
        # avoids creating a second Trackable path through `_all_variables`,
        # which previously caused each variable to be serialized twice.
        fns = [self._get_concrete_fn(name) for name in self._endpoint_names]
        tvs, ntvs = _list_variables_used_by_fns(fns)
        self._tf_trackable.trainable_variables = list(tvs)
        self._tf_trackable.non_trainable_variables = list(ntvs)
        self._tf_trackable.variables = list(tvs + ntvs)
        self._track_lookup_tables_and_misc_assets()
