import tensorflow as tf

from keras.src import tree
from keras.src.backend.tensorflow.trackable import KerasAutoTrackable
from keras.src.utils import tf_utils
from keras.src.utils import tracking


class TFLayer(KerasAutoTrackable):
    def __init__(self, *args, **kwargs):
        # Export-related attributes
        self._saved_model_inputs_spec = None
        self._saved_model_arg_spec = None
        self._tracked = []

    def _set_save_spec(self, inputs, args=None, kwargs=None):
        """Defines the save spec so that serialization can trace layer calls.

        The TensorSpecs of the call function `inputs`, `args`, and `kwargs` are
        saved into a tuple of `([inputs] + args, kwargs)`.

        Args:
          inputs: possibly nested inputs passed into the call function.
          args: a list of positional arguments passed into call.
          kwargs: a dictionary of keyword arguments passed into call.
        """
        if self._saved_model_inputs_spec is not None:
            return  # Already set.

        inputs_spec = tree.map_structure(tf_utils.get_tensor_spec, inputs)
        args_spec = tree.map_structure(tf_utils.get_tensor_spec, args or [])
        kwargs_spec = {}
        # Filter out non-tensor arguments from kwargs.
        for key, kwarg in kwargs.items():
            flat_kwarg = tree.flatten(kwarg)
            flat_specs = [tf_utils.get_tensor_spec(x) for x in flat_kwarg]
            if any(s is None for s in flat_specs):
                continue
            kwargs_spec[key] = tree.pack_sequence_as(kwarg, flat_specs)

        self._saved_model_inputs_spec = inputs_spec
        self._saved_model_arg_spec = (
            [inputs_spec] + list(args_spec),
            kwargs_spec,
        )

    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def _trackable_children(self, save_type="checkpoint", **kwargs):
        if save_type == "savedmodel":
            # SavedModel needs to ignore the execution functions.
            train_function = getattr(self, "train_function", None)
            test_function = getattr(self, "test_function", None)
            predict_function = getattr(self, "predict_function", None)
            self.train_function = None
            self.test_function = None
            self.predict_function = None

        children = super()._trackable_children(save_type, **kwargs)

        if save_type == "savedmodel":
            self.train_function = train_function
            self.test_function = test_function
            self.predict_function = predict_function

            # Convert Keras tracked collections to plain Python structures
            # without creating TensorFlow trackable dependencies
            self._convert_tracked_collections(children)

        return children

    def _convert_tracked_collections(self, children):
        """Convert TrackedList/Dict/Set to plain Python structures."""
        for tracked_attr in self._tracked:
            tracked_item = getattr(self, tracked_attr)
            if isinstance(tracked_item, tracking.TrackedList):
                children[tracked_attr] = list(tracked_item)
            if isinstance(tracked_item, tracking.TrackedDict):
                children[tracked_attr] = dict(tracked_item)
            if isinstance(tracked_item, tracking.TrackedSet):
                children[tracked_attr] = list(tracked_item)

    def _get_save_spec(self, dynamic_batch=True):
        """Compatibility shim for TensorFlow saving utilities.

        TensorFlow's SavedModel / TFLite export paths (e.g.,
        tf.lite.TFLiteConverter.from_keras_model) expect a `_get_save_spec`
        method on models. This method generates TensorSpec objects
        describing the model's input signature.

        Args:
            dynamic_batch: whether to set the batch dimension to `None`.

        Returns:
            A TensorSpec, list or dict mirroring the model inputs, or
            `None` when specs cannot be inferred.
        """
        # Lazy import to avoid circular dependency
        from keras.src.export.export_utils import make_tf_tensor_spec

        # Fall back to building specs from `self.inputs`
        inputs = getattr(self, "inputs", None)
        if inputs is None:
            return None

        return tree.map_structure(
            lambda x: make_tf_tensor_spec(x, dynamic_batch=dynamic_batch),
            inputs,
        )

    @property
    def _default_save_signature(self):
        """For SavedModel support: returns the default serving signature."""

        from keras.src.models.functional import Functional
        from keras.src.models.model import Model
        from keras.src.models.sequential import Sequential

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
            input_signature = (
                tree.map_structure(
                    lambda x: tf.TensorSpec(x.shape, x.dtype), inputs
                ),
            )
        else:
            input_signature = tuple(
                tree.map_shape_structure(
                    lambda s: tf.TensorSpec(s, self.input_dtype), value
                )
                for value in self._build_shapes_dict.values()
            )

        @tf.function(input_signature=input_signature)
        def serving_default(inputs):
            return self(inputs)

        return serving_default
