import tensorflow as tf
from keras.src.export.export_utils import get_input_signature, make_tf_tensor_spec
from keras import tree
import tempfile
import os

class GenerationModule(tf.Module):
    """
    A tf.Module to wrap a CausalLM for exporting to TFLite with separate
    prompt processing and token generation functions.
    """
    def __init__(self, model, max_sequence_length):
        super().__init__()
        self.model = model
        self.max_sequence_length = max_sequence_length

    @tf.function
    def initialize(self, token_ids, padding_mask):
        """
        Initializes the key/value cache by processing the input prompt.

        This function creates an empty cache and then "seeds" it by running a
        forward pass on the prompt. This is equivalent to the `_build_cache`
        logic in KerasNLP's CausalLM models.

        Args:
            token_ids: A tf.Tensor of shape [batch_size, seq_len].
            padding_mask: A tf.Tensor of shape [batch_size, seq_len].

        Returns:
            A dictionary containing the `initial_cache`.
        """
        backbone = self.model.backbone
        batch_size = tf.shape(token_ids)[0]
        cache_shape = [
            batch_size,
            backbone.num_layers,
            2,  # For key and value
            self.max_sequence_length,
            backbone.num_key_value_heads,
            backbone.head_dim,
        ]
        # Create an empty cache with a static max_sequence_length.
        cache = tf.zeros(cache_shape, dtype=self.model.compute_dtype)

        # Seed the cache by calling call_with_cache on the prompt.
        # The cache_update_index is 0, and the tokens are the whole prompt.
        # We only need the resulting cache.
        _, _, seeded_cache = self.model.call_with_cache(
            token_ids, cache, cache_update_index=0
        )
        return {"initial_cache": seeded_cache}

    @tf.function
    def decode(self, token_ids, cache, cache_update_index):
        """
        Performs one decoding step to generate the next token.

        Args:
            token_ids: The current token, shape [batch_size, 1].
            cache: The key/value cache from the previous step.
            cache_update_index: The index at which to update the cache.

        Returns:
            A dictionary containing the `logits` and the `updated_cache`.
        """
        logits, _, new_cache = self.model.call_with_cache(
            token_ids, cache, cache_update_index
        )
        return {"logits": logits, "updated_cache": new_cache}


class LiteRTExporter:
    """
    Exporter for the LiteRT (TFLite) format.
    """
    
    def __init__(self, model, input_signature=None, verbose=None, max_sequence_length=512, **kwargs):
        self.model = model
        self.input_signature = input_signature
        self.verbose = verbose or 0
        self.max_sequence_length = max_sequence_length
        self.kwargs = kwargs

    def export(self, filepath):
        if self.verbose:
            print("Starting LiteRT export...")
        
        tflite_model = self._convert_to_tflite()
        
        if self.verbose:
            print(f"LiteRT model converted successfully. Size: {len(tflite_model)} bytes")
        
        # Step 4: Save the model to the specified file path.
        if not filepath.endswith('.tflite'):
            filepath = filepath + '.tflite'
        
        with open(filepath, "wb") as f:
            f.write(tflite_model)
        
        if self.verbose:
            print(f"Exported model to {filepath}")

    def _convert_to_tflite(self):
        """
        Converts a Keras model to TFLite, automatically selecting the best
        conversion path based on the model's architecture.
        """
        # Use duck-typing to check for a Keras-Hub style CausalLM model
        # that supports efficient, cached generation.
        is_generative = hasattr(self.model, "call_with_cache") and hasattr(
            self.model.backbone, "num_key_value_heads"
        )

        if is_generative:
            if self.verbose:
                print(
                    "Generative CausalLM model detected. Exporting with "
                    "'initialize' and 'decode' signatures for efficient generation."
                )
            return self._convert_generative_model()
        else:
            if self.verbose:
                print(
                    "Standard model detected. Using direct conversion path "
                    "with a single 'serving_default' signature."
                )
            # Fallback to the standard conversion for non-generative models.
            if self.input_signature is None:
                self.input_signature = get_input_signature(
                    self.model, self.max_sequence_length
                )
            concrete_fn = self._get_concrete_function(self.input_signature)
            return self._convert_direct(concrete_fn)

    def _get_attribute(self, obj, attribute_names):
        """Safely get an attribute from a list of possible names."""
        for name in attribute_names:
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(
            f"Could not find any of the following attributes on object "
            f"'{obj.__class__.__name__}': {', '.join(attribute_names)}"
        )

    def _convert_generative_model(self):
        """
        Exports a CausalLM model via SavedModel with two distinct signatures
        for 'initialize' (prompt processing) and 'decode' (token generation).
        """
        module = GenerationModule(self.model, self.max_sequence_length)
        backbone = self.model.backbone

        # Use the helper to generically get attributes
        num_layers = self._get_attribute(backbone, ["num_layers"])
        num_key_value_heads = self._get_attribute(
            backbone, ["num_key_value_heads", "num_attention_heads"]
        )
        head_dim = self._get_attribute(backbone, ["head_dim", "key_dim"])

        # 1. Define the TensorSpec for the 'initialize' signature.
        init_signature = (
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="token_ids"),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="padding_mask"),
        )

        # 2. Define the TensorSpec for the 'decode' signature.
        cache_shape = [
            None,  # batch_size
            num_layers,
            2,  # key and value
            self.max_sequence_length,
            num_key_value_heads,
            head_dim,
        ]
        decode_signature = (
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32, name="token_ids"),
            tf.TensorSpec(shape=cache_shape, dtype=self.model.compute_dtype, name="cache"),
            tf.TensorSpec(shape=[], dtype=tf.int32, name="cache_update_index"),
        )

        # 3. Get concrete functions and bundle them as signatures.
        signatures = {
            "initialize": module.initialize.get_concrete_function(*init_signature),
            "decode": module.decode.get_concrete_function(*decode_signature),
        }

        # 4. Save as a SavedModel and then convert to TFLite.
        with tempfile.TemporaryDirectory() as tmpdir:
            if self.verbose:
                print(f"Saving temporary SavedModel to {tmpdir}")
            tf.saved_model.save(module, tmpdir, signatures=signatures)
            
            converter = tf.lite.TFLiteConverter.from_saved_model(tmpdir)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            # This is crucial for including both functions in the TFLite model.
            converter.signature_keys = ["initialize", "decode"]
            tflite_model = converter.convert()
        
        return tflite_model

    def _get_concrete_function(self, input_signature):
        """Create a tf.function and get its concrete function (for non-generative models)."""
        if isinstance(input_signature, dict):
            tf_signature = [make_tf_tensor_spec(spec) for spec in input_signature.values()]
            input_keys = list(input_signature.keys())
            
            @tf.function(input_signature=tf_signature)
            def model_fn(*inputs):
                input_dict = {key: tensor for key, tensor in zip(input_keys, inputs)}
                return self.model(input_dict)
        
        elif isinstance(input_signature, list):
            tf_signature = [make_tf_tensor_spec(spec) for spec in input_signature]
            
            @tf.function(input_signature=tf_signature)
            def model_fn(*inputs):
                return self.model(list(inputs))
        
        else: # Assumes a single tensor or spec
            tf_signature = [make_tf_tensor_spec(input_signature)]
            
            @tf.function(input_signature=tf_signature)
            def model_fn(input_tensor):
                return self.model(input_tensor)
        
        return model_fn.get_concrete_function()

    def _convert_direct(self, concrete_fn):
        """Directly convert a concrete function to TFLite (for non-generative models)."""
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [concrete_fn], self.model
        )
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        return converter.convert()