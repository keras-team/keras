import keras
from keras.src.api_export import keras_export


@keras_export("keras.distillation.BaseDistillationStrategy")
class BaseDistillationStrategy:
    """Base class for distillation strategies.

    Distillation strategies define how to compute the distillation loss
    between teacher and student outputs.
    To create custom distillation strategies, subclass this class and
    override the compute_loss method.
    """

    def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
        """Compute distillation loss between teacher and student outputs.
        Args:
            teacher_outputs: Outputs from the teacher model. Can be a single
                tensor or a list/tuple of tensors for multi-output models.
            student_outputs: Outputs from the student model. Can be a single
                tensor or a list/tuple of tensors for multi-output models.
            **kwargs: Additional arguments for custom strategies.
        Returns:
            Distillation loss tensor.
        """
        raise NotImplementedError("Subclasses must implement compute_loss")

    def validate_outputs(self, teacher_outputs, student_outputs):
        """Validate that teacher and student outputs are compatible.

        Args:
            teacher_outputs: Outputs from the teacher model.
            student_outputs: Outputs from the student model.
        Raises:
            ValueError: If outputs are not compatible.
        """
        # Default implementation - can be overridden by subclasses
        if not isinstance(teacher_outputs, (list, tuple)):
            teacher_outputs = [teacher_outputs]
        if not isinstance(student_outputs, (list, tuple)):
            student_outputs = [student_outputs]

        if len(teacher_outputs) != len(student_outputs):
            raise ValueError(
                f"Teacher and student must have the same number of outputs. "
                f"Teacher has {len(teacher_outputs)} outputs, "
                f"student has {len(student_outputs)} outputs."
            )


@keras_export("keras.distillation.LogitsDistillation")
class LogitsDistillation(BaseDistillationStrategy):
    """Logits distillation strategy using Keras built-in loss functions.

    This strategy distills knowledge using the logits (pre-softmax outputs)
    from teacher and student models.

    Args:
        temperature: Temperature for softmax scaling. Higher values produce
            softer probability distributions. If None, will use the default
            temperature from the Distiller. Defaults to None.
        loss_type: Type of loss function to use. Options:
            - "kl_divergence": KL divergence using keras.losses.kl_divergence
            - "categorical_crossentropy": Categorical crossentropy using
              keras.losses.categorical_crossentropy
        output_index: Index of the output to use for multi-output models.
            Defaults to 0.
    """

    def __init__(
        self, temperature=None, loss_type="kl_divergence", output_index=0
    ):
        # If no temperature provided, use sentinel value for Distiller detection
        self.temperature = temperature if temperature is not None else 3.0
        self._temperature_explicitly_set = temperature is not None
        self.loss_type = loss_type
        self.output_index = output_index

        # Validate loss_type
        valid_loss_types = ["kl_divergence", "categorical_crossentropy"]
        if loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}")

    def set_default_temperature(self, default_temperature):
        """Set the default temperature if none was explicitly provided."""
        if not self._temperature_explicitly_set:
            self.temperature = default_temperature

    def validate_outputs(self, teacher_outputs, student_outputs):
        """Validate that outputs are compatible for logits distillation."""
        super().validate_outputs(teacher_outputs, student_outputs)

        # Ensure outputs are lists/tuples
        if not isinstance(teacher_outputs, (list, tuple)):
            teacher_outputs = [teacher_outputs]
        if not isinstance(student_outputs, (list, tuple)):
            student_outputs = [student_outputs]

        # Check output index is valid
        if self.output_index >= len(teacher_outputs):
            raise ValueError(
                f"output_index {self.output_index} is out of range. "
                f"Teacher has {len(teacher_outputs)} outputs."
            )
        if self.output_index >= len(student_outputs):
            raise ValueError(
                f"output_index {self.output_index} is out of range. "
                f"Student has {len(student_outputs)} outputs."
            )

        # Check that the selected outputs have compatible shapes
        teacher_output = teacher_outputs[self.output_index]
        student_output = student_outputs[self.output_index]

        if teacher_output.shape[-1] != student_output.shape[-1]:
            raise ValueError(
                f"Teacher and student outputs must have the same number of "
                f"classes. "
                f"Teacher output shape: {teacher_output.shape}, "
                f"Student output shape: {student_output.shape}"
            )

    def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
        """Compute distillation loss using Keras built-in loss functions.

        Args:
            teacher_outputs: Logits from teacher model. Can be a single tensor
                or a list/tuple of tensors for multi-output models.
            student_outputs: Logits from student model. Can be a single tensor
                or a list/tuple of tensors for multi-output models.
            **kwargs: Additional arguments (ignored).
        Returns:
            Distillation loss tensor.
        """
        from keras import ops

        # Normalize outputs to lists
        if not isinstance(teacher_outputs, (list, tuple)):
            teacher_outputs = [teacher_outputs]
        if not isinstance(student_outputs, (list, tuple)):
            student_outputs = [student_outputs]

        # Get the outputs to distill
        teacher_logits = teacher_outputs[self.output_index]
        student_logits = student_outputs[self.output_index]

        # Apply temperature scaling
        teacher_logits = teacher_logits / self.temperature
        student_logits = student_logits / self.temperature

        if self.loss_type == "kl_divergence":
            # Convert to probabilities for KL divergence
            teacher_probs = ops.softmax(teacher_logits, axis=-1)
            student_probs = ops.softmax(student_logits, axis=-1)

            # Use Keras KLDivergence directly and reduce to scalar
            loss = ops.mean(
                keras.losses.kl_divergence(teacher_probs, student_probs)
            )

        elif self.loss_type == "categorical_crossentropy":
            # Convert teacher to probabilities, keep student as logits
            teacher_probs = ops.softmax(teacher_logits, axis=-1)

            # Use Keras CategoricalCrossentropy directly and reduce to scalar
            loss = ops.mean(
                keras.losses.categorical_crossentropy(
                    teacher_probs, student_logits
                )
            )

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Scale by temperature^2 for consistency with literature
        return loss * (self.temperature**2)

    def get_config(self):
        """Get configuration for serialization."""
        return {
            "temperature": self.temperature,
            "loss_type": self.loss_type,
            "output_index": self.output_index,
        }


@keras_export("keras.distillation.FeatureDistillation")
class FeatureDistillation(BaseDistillationStrategy):
    """Feature distillation strategy using intermediate layer features.

    This strategy distills intermediate features from teacher to student,
    not just the final outputs. It creates feature extraction models
    to extract outputs from specified intermediate layers.

    Note: If teacher and student features have different shapes, you may need
    to add alignment layers or use models with compatible intermediate
    feature dimensions.

    Args:
        loss_type: Type of loss function to use. Options:
            - "mse": Mean squared error using keras.losses.mean_squared_error
            - "cosine": Cosine similarity using keras.losses.cosine_similarity
        teacher_layer_name: Name of the teacher layer to extract features from.
            If None, uses the final output. Defaults to None.
        student_layer_name: Name of the student layer to extract features from.
            If None, uses the final output. Defaults to None.
    """

    def __init__(
        self, loss_type="mse", teacher_layer_name=None, student_layer_name=None
    ):
        self.loss_type = loss_type
        self.teacher_layer_name = teacher_layer_name
        self.student_layer_name = student_layer_name

        # Feature extraction models (created when needed)
        self._teacher_feature_model = None
        self._student_feature_model = None

        # Validate loss_type
        valid_loss_types = ["mse", "cosine"]
        if loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}")

    def _get_teacher_features(self, teacher_model, inputs):
        """Extract features from teacher model."""
        if self.teacher_layer_name is None:
            # No specific layer, use the full model
            return teacher_model(inputs, training=False)

        # For intermediate layer extraction, we need to create a custom function
        # that extracts the output at the specified layer
        if self._teacher_feature_model is None:
            self._teacher_feature_model = self._create_feature_extractor(
                teacher_model, self.teacher_layer_name
            )

        return self._teacher_feature_model(inputs, training=False)

    def _get_student_features(self, student_model, inputs):
        """Extract features from student model."""
        if self.student_layer_name is None:
            # No specific layer, use the full model
            return student_model(inputs, training=True)

        # For intermediate layer extraction, we need to create a custom function
        # that extracts the output at the specified layer
        if self._student_feature_model is None:
            self._student_feature_model = self._create_feature_extractor(
                student_model, self.student_layer_name
            )

        return self._student_feature_model(inputs, training=True)

    def _create_feature_extractor(self, model, layer_name):
        """Create a feature extractor function for the specified layer.

        Args:
            model: The model to extract features from.
            layer_name: Name of the layer to extract features from.
                       If None, returns the original model.

        Returns:
            A callable that extracts features from the specified layer.
        """
        if layer_name is None:
            # Return the original model if no layer specified
            return model

        # Find the layer by name
        target_layer = None
        layer_index = None
        for i, layer in enumerate(model.layers):
            if layer.name == layer_name:
                target_layer = layer
                layer_index = i
                break

        if target_layer is None:
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"Available layers: {[layer.name for layer in model.layers]}"
            )

        # Create a custom model class that extracts intermediate features
        class FeatureExtractor(keras.Model):
            def __init__(self, original_model, target_layer_index):
                super().__init__(
                    name=f"{original_model.name}_features_{layer_name}"
                )
                self.original_model = original_model
                self.target_layer_index = target_layer_index

            def call(self, inputs, training=None):
                # Run through the model up to the target layer
                x = inputs
                for i, layer in enumerate(self.original_model.layers):
                    x = layer(x, training=training)
                    if i == self.target_layer_index:
                        return x
                return x  # Fallback, shouldn't reach here

        return FeatureExtractor(model, layer_index)

    def validate_outputs(self, teacher_outputs, student_outputs):
        """Validate that outputs are compatible for feature distillation."""
        super().validate_outputs(teacher_outputs, student_outputs)

        # For feature distillation, we need to ensure the features have
        # compatible shapes for the chosen loss function
        if not isinstance(teacher_outputs, (list, tuple)):
            teacher_outputs = [teacher_outputs]
        if not isinstance(student_outputs, (list, tuple)):
            student_outputs = [student_outputs]

        # Basic shape compatibility check
        teacher_features = teacher_outputs[0]  # Use first output by default
        student_features = student_outputs[0]  # Use first output by default

        if len(teacher_features.shape) != len(student_features.shape):
            raise ValueError(
                f"Teacher and student features must have the same number of "
                f"dimensions. "
                f"Teacher shape: {teacher_features.shape}, "
                f"Student shape: {student_features.shape}"
            )

        # For MSE loss, shapes must match exactly
        if self.loss_type == "mse":
            if teacher_features.shape != student_features.shape:
                raise ValueError(
                    f"For MSE loss, teacher and student features must have "
                    f"identical shapes. Got teacher: {teacher_features.shape}, "
                    f"student: {student_features.shape}. "
                    f"Consider using 'cosine' loss type for different sizes "
                    f"or add alignment layers to make features compatible."
                )

        # For cosine loss, only last dimension needs to match (features)
        elif self.loss_type == "cosine":
            if teacher_features.shape[-1] != student_features.shape[-1]:
                raise ValueError(
                    f"For cosine similarity loss, teacher and student features "
                    f"must have the same feature dimension (last axis). "
                    f"Got teacher: {teacher_features.shape[-1]}, "
                    f"student: {student_features.shape[-1]}. "
                    f"Consider adding a projection layer to align dimensions."
                )

    def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
        """Compute feature distillation loss using extracted features.

        Note: This method expects the outputs to already be the extracted
        features from the specified layers, not the final model outputs.
        The Distiller class is responsible for extracting the features
        using the methods provided by this strategy.

        Args:
            teacher_outputs: Intermediate features from teacher model.
                Can be a single tensor or a list/tuple of tensors.
            student_outputs: Intermediate features from student model.
                Can be a single tensor or a list/tuple of tensors.
            **kwargs: Additional arguments (ignored).
        Returns:
            Feature distillation loss tensor.
        """
        from keras import ops

        # Normalize outputs to lists
        if not isinstance(teacher_outputs, (list, tuple)):
            teacher_outputs = [teacher_outputs]
        if not isinstance(student_outputs, (list, tuple)):
            student_outputs = [student_outputs]

        # Use first output by default (can be extended to use specific outputs)
        teacher_features = teacher_outputs[0]
        student_features = student_outputs[0]

        if self.loss_type == "mse":
            # Use Keras MeanSquaredError directly and reduce to scalar
            return ops.mean(
                keras.losses.mean_squared_error(
                    teacher_features, student_features
                )
            )

        elif self.loss_type == "cosine":
            # Use Keras CosineSimilarity directly (returns similarity, convert
            # to distance)
            similarity = ops.mean(
                keras.losses.cosine_similarity(
                    teacher_features, student_features
                )
            )
            # Convert similarity to distance: distance = 1 - similarity
            return 1.0 - similarity

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def get_config(self):
        """Get configuration for serialization."""
        return {
            "loss_type": self.loss_type,
            "teacher_layer_name": self.teacher_layer_name,
            "student_layer_name": self.student_layer_name,
        }


@keras_export("keras.distillation.MultiOutputDistillation")
class MultiOutputDistillation(BaseDistillationStrategy):
    """Multi-output distillation strategy.

    Multi-output distillation strategy applies distillation to multiple
    outputs. This strategy allows different distillation strategies to be
    applied to different outputs of multi-output models.

    Args:
        output_strategies: Dict mapping output indices to distillation
            strategies.
            Each strategy will be applied to the corresponding output.
        weights: Dict mapping output indices to weights for combining losses.
            If None, all outputs are weighted equally. Defaults to None.
    """

    def __init__(self, output_strategies, weights=None):
        self.output_strategies = output_strategies
        self.weights = weights or {idx: 1.0 for idx in output_strategies.keys()}

    def validate_outputs(self, teacher_outputs, student_outputs):
        """Validate outputs are compatible for multi-output distillation."""
        super().validate_outputs(teacher_outputs, student_outputs)

        # Ensure outputs are lists/tuples
        if not isinstance(teacher_outputs, (list, tuple)):
            teacher_outputs = [teacher_outputs]
        if not isinstance(student_outputs, (list, tuple)):
            student_outputs = [student_outputs]

        # Check that all required outputs exist
        max_output_index = max(self.output_strategies.keys())
        if max_output_index >= len(teacher_outputs):
            raise ValueError(
                f"Teacher model doesn't have enough outputs. "
                f"Required: {max_output_index + 1}, available: "
                f"{len(teacher_outputs)}"
            )
        if max_output_index >= len(student_outputs):
            raise ValueError(
                f"Student model doesn't have enough outputs. "
                f"Required: {max_output_index + 1}, available: "
                f"{len(student_outputs)}"
            )

        # Validate each strategy with its corresponding outputs
        for output_idx, strategy in self.output_strategies.items():
            if hasattr(strategy, "validate_outputs"):
                strategy.validate_outputs(
                    [teacher_outputs[output_idx]], [student_outputs[output_idx]]
                )

    def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
        """Compute multi-output distillation loss.

        Args:
            teacher_outputs: Outputs from teacher model.
            student_outputs: Outputs from student model.
            **kwargs: Additional arguments passed to individual strategies.

        Returns:
            Combined distillation loss tensor.
        """
        # Normalize outputs to lists
        if not isinstance(teacher_outputs, (list, tuple)):
            teacher_outputs = [teacher_outputs]
        if not isinstance(student_outputs, (list, tuple)):
            student_outputs = [student_outputs]

        total_loss = 0.0

        for output_idx, strategy in self.output_strategies.items():
            teacher_output = teacher_outputs[output_idx]
            student_output = student_outputs[output_idx]

            # Compute loss for this output
            output_loss = strategy.compute_loss(
                [teacher_output], [student_output], **kwargs
            )

            # Apply weight
            weight = self.weights.get(output_idx, 1.0)
            total_loss += weight * output_loss

        return total_loss

    def get_config(self):
        """Get configuration for serialization."""
        return {
            "output_strategies": self.output_strategies,
            "weights": self.weights,
        }
