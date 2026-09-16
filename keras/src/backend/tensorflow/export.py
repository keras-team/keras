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
        # Under the TensorFlow backend, endpoint functions capture the
        # `tf.Variable` objects associated with Keras `Variable` wrappers
        # registered through `track()`. These captured variables must be
        # tracked directly TensorFlow recognizes them as trackable during
        # SavedModel export.
        #
        # Compare the captured TensorFlow variables with the underlying `_value`
        # of the originally tracked Keras variables to avoid adding the
        # same resource through two Trackable paths. Variables
        # that were explicitly tracked but
        # are not captured by any endpoint are preserved.
        #
        # `_all_variables` is updated from the final deduplicated collections
        # so that the same variables are not serialized twice while
        # remaining available for downstream consumers such as LiteRT.
        fns = [self._get_concrete_fn(name) for name in self._endpoint_names]
        tvs, ntvs = _list_variables_used_by_fns(fns)

        captured_ids = {id(v) for v in tvs + ntvs}

        original_trainable = list(self._tf_trackable.trainable_variables)
        original_non_trainable = list(
            self._tf_trackable.non_trainable_variables
        )
        extra_trainable = [
            v
            for v in original_trainable
            if id(getattr(v, "_value", v)) not in captured_ids
        ]
        extra_non_trainable = [
            v
            for v in original_non_trainable
            if id(getattr(v, "_value", v)) not in captured_ids
        ]
        self._tf_trackable.trainable_variables = list(tvs) + extra_trainable
        self._tf_trackable.non_trainable_variables = (
            list(ntvs) + extra_non_trainable
        )
        self._tf_trackable.variables = (
            self._tf_trackable.trainable_variables
            + self._tf_trackable.non_trainable_variables
        )
        self._tf_trackable._all_variables = self._tf_trackable.variables

        self._track_lookup_tables_and_misc_assets()
