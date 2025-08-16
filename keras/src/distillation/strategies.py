import keras
from keras.src.api_export import keras_export


@keras_export("keras.distillation.BaseDistillationStrategy")
class BaseDistillationStrategy:
    """Base class for distillation strategies.

    Distillation strategies define how to compute the distillation loss
    between teacher and student outputs. Each strategy implements a specific
    approach to knowledge transfer, from simple logits matching to multi-output
    distillation.

    To create custom distillation strategies, subclass this class and
    override the `compute_loss` method.
    """

    def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
        """Compute distillation loss between teacher and student outputs.

        This method should implement the specific distillation logic for
        transferring knowledge from teacher to student.

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

        This method ensures that the outputs from teacher and student models
        are compatible for the specific distillation strategy. It should check
        shapes, dimensions, and other requirements.

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
    """Distillation strategy that transfers knowledge from final model outputs.

    This strategy applies temperature scaling to the teacher's logits before
    computing the loss between teacher and student predictions. It's the most
    common approach for knowledge distillation.

    How Logits Distillation Works:

    1. Temperature Scaling: The teacher's logits are divided by a `temperature`
       parameter (typically 3-5) before applying softmax. This creates "softer"
       probability distributions that reveal relationships between classes.

    2. Loss Computation: The loss is computed between the temperature-scaled
       teacher logits and student logits using either KL divergence or
       categorical crossentropy.

    When to Use Logits Distillation:

    - General Classification: Works well for most classification tasks
    - Model Compression: Effective for reducing model size while maintaining
      accuracy
    - Transfer Learning: Good for leveraging knowledge from pre-trained models
    - Ensemble Distillation: Can combine multiple teacher models

    Temperature Guidelines:

    - Low Temperature (1-2): Sharp distributions, similar to hard labels
    - Medium Temperature (3-5): Balanced softness, most commonly used
    - High Temperature (6-10): Very soft distributions, reveals subtle
      relationships

    Args:
        temperature: Temperature for softmax scaling. Higher values produce
            softer probability distributions that are easier for the student to
            learn. Typical values range from 3-5. Defaults to 3.0.
        loss_type: Type of loss function to use. Options:
            - `"kl_divergence"`: KL divergence between teacher and student
              distributions
            - `"categorical_crossentropy"`: Crossentropy with teacher as target
        output_index: Index of the output to use for multi-output models.
            Defaults to 0.

    Example:

    ```python
    # Basic logits distillation
    strategy = LogitsDistillation(temperature=3.0)

    # With categorical crossentropy loss
    strategy = LogitsDistillation(
        temperature=4.0,
        loss_type="categorical_crossentropy"
    )

    # Custom loss by subclassing
    class CustomLogitsDistillation(LogitsDistillation):
        def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
            from keras import ops
            # Get the outputs to distill
            teacher_logits = teacher_outputs[self.output_index]
            student_logits = student_outputs[self.output_index]

            # Apply temperature scaling
            teacher_logits = teacher_logits / self.temperature
            student_logits = student_logits / self.temperature

            # Custom loss computation
            teacher_probs = ops.softmax(teacher_logits, axis=-1)
            student_probs = ops.softmax(student_logits, axis=-1)
            return ops.mean(
                keras.losses.kl_divergence(teacher_probs, student_probs)
            )

    strategy = CustomLogitsDistillation(temperature=3.0)

    # For multi-output models
    strategy = LogitsDistillation(
        temperature=3.0,
        output_index=1  # Use second output
    )
    ```
    """

    def __init__(
        self,
        temperature=3.0,
        loss_type="kl_divergence",
        output_index=0,
    ):
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        self.output_index = output_index

        # Validate loss_type
        if loss_type not in ["kl_divergence", "categorical_crossentropy"]:
            raise ValueError(
                f"loss_type must be one of ['kl_divergence', "
                f"'categorical_crossentropy'], got {loss_type}"
            )

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

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        return cls(**config)


@keras_export("keras.distillation.FeatureDistillation")
class FeatureDistillation(BaseDistillationStrategy):
    """Feature distillation strategy using intermediate layer representations.

    Feature distillation transfers knowledge from intermediate layers of the
    teacher model to corresponding layers of the student model. This approach
    helps the student learn better internal representations and often leads
    to better performance compared to logits-only distillation.

    How Feature Distillation Works:

    1. Layer Selection: Specify which intermediate layers from teacher and
       student models to use for distillation. These layers should have
       compatible architectures or similar semantic meaning.

    2. Feature Extraction: Extract activations from the specified layers
       during forward pass. The teacher features are computed with
       `training=False` (frozen), while student features are computed with
       `training=True`.

    3. Loss Computation: Compute loss between teacher and student features
       using either MSE (for identical shapes) or cosine similarity (for
       different shapes).

    When to Use Feature Distillation:

    - Similar Architectures: When teacher and student have similar layer
      structures (e.g., both are CNNs with similar depths)
    - Performance Improvement: Often leads to better student performance
      than logits-only distillation
    - Representation Learning: Helps student learn better internal features
    - Multi-Scale Distillation: Can distill features from multiple layers
      simultaneously

    Layer Selection Guidelines:

    - Early Layers: Capture low-level features (edges, textures)
    - Middle Layers: Capture mid-level features (shapes, patterns)
    - Late Layers: Capture high-level features (semantic concepts)
    - Compatible Sizes: Choose layers with similar output dimensions
    - Semantic Alignment: Match layers that serve similar functions

    Loss Type Selection:

    - `"mse"`: Use when teacher and student features have identical shapes.
      Provides direct feature matching.
    - `"cosine"`: Use when features have different shapes but
      same feature dimension (last axis). Focuses on feature direction
      rather than magnitude.

    Args:
        loss_type: Type of loss function to use. Options:
            - `"mse"`: Mean squared error between teacher and student features
            - `"cosine"`: Cosine similarity between feature vectors
        teacher_layer_name: Name of the teacher layer to extract features from.
            If None, uses the final output. Defaults to None.
        student_layer_name: Name of the student layer to extract features from.
            If None, uses the final output. Defaults to None.

    Examples:

    ```python
    # Basic feature distillation from final outputs
    strategy = FeatureDistillation(loss_type="mse")

    # Custom loss by subclassing
    class CustomFeatureDistillation(FeatureDistillation):
        def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
            from keras import ops
            # Use first output by default
            teacher_features = teacher_outputs[0]
            student_features = student_outputs[0]

            # Custom L1 loss for feature distillation
            return ops.mean(ops.abs(teacher_features - student_features))

    strategy = CustomFeatureDistillation(
        teacher_layer_name="dense_1",
        student_layer_name="dense_1"
    )

    # Distill from specific layers with compatible shapes
    strategy = FeatureDistillation(
        loss_type="mse",
        teacher_layer_name="dense_1",
        student_layer_name="dense_1"
    )

    # Use cosine similarity for different feature sizes
    strategy = FeatureDistillation(
        loss_type="cosine",
        teacher_layer_name="conv2d_2",
        student_layer_name="conv2d_1"
    )

    # Distill from final outputs (equivalent to logits distillation)
    strategy = FeatureDistillation(
        loss_type="mse",
        teacher_layer_name=None,  # Final output
        student_layer_name=None   # Final output
    )
    ```
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
        # No specific layer, use the final model output
        if self.teacher_layer_name is None:
            return teacher_model(inputs, training=False)

        # For intermediate layer extraction, we need to create a custom function
        # that extracts the output at the specified layer
        if self._teacher_feature_model is None:
            # Build the model first if needed (for Sequential models)
            try:
                self._teacher_feature_model = self._create_feature_extractor(
                    teacher_model, self.teacher_layer_name
                )
            except ValueError as e:
                if "no defined inputs" in str(e).lower():
                    # Build the model by calling it with the inputs first
                    _ = teacher_model(inputs, training=False)
                    # Now try again
                    self._teacher_feature_model = (
                        self._create_feature_extractor(
                            teacher_model, self.teacher_layer_name
                        )
                    )
                else:
                    raise

        return self._teacher_feature_model(inputs, training=False)

    def _get_student_features(self, student_model, inputs):
        """Extract features from student model."""
        # No specific layer, use the final model output
        if self.student_layer_name is None:
            return student_model(inputs, training=True)

        # For intermediate layer extraction, we need to create a custom function
        # that extracts the output at the specified layer
        if self._student_feature_model is None:
            # Build the model first if needed (for Sequential models)
            try:
                self._student_feature_model = self._create_feature_extractor(
                    student_model, self.student_layer_name
                )
            except ValueError as e:
                if "no defined inputs" in str(e).lower():
                    # Build the model by calling it with the inputs first
                    _ = student_model(inputs, training=True)
                    # Now try again
                    self._student_feature_model = (
                        self._create_feature_extractor(
                            student_model, self.student_layer_name
                        )
                    )
                else:
                    raise

        return self._student_feature_model(inputs, training=True)

    def _create_feature_extractor(self, model, layer_name):
        """Create a feature extractor function for the specified layer.

        Args:
            model: The model to extract features from.
            layer_name: Name of the layer to extract features from.
                       If None, returns the original model.

        Returns:
            A keras.Model that extracts features from the specified layer.
        """
        if layer_name is None:
            # Return the original model if no layer specified
            return model

        # Find the layer by name
        target_layer = None
        for layer in model.layers:
            if layer.name == layer_name:
                target_layer = layer
                break

        if target_layer is None:
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"This may happen with a subclassed model that cannot be "
                f"traversed using the standard layer API. "
                f"Available layers: {[layer.name for layer in model.layers]}"
            )

        # Create a new model that extracts features from the specified layer.
        # This approach is robust for models created with the Functional API.
        try:
            return keras.Model(
                inputs=model.inputs,
                outputs=target_layer.output,
                name=f"{model.name}_features_{layer_name}",
            )
        except (ValueError, AttributeError) as e:
            # Handle the case where the model doesn't have defined inputs yet
            # (common with Sequential models that haven't been built)
            error_msg = str(e).lower()
            if (
                "no defined inputs" in error_msg
                or "has no defined inputs" in error_msg
            ):
                raise ValueError(
                    f"Model '{model.name}' has no defined inputs yet. "
                    f"Please call the model with some input data first to "
                    f"build it, or use the Functional API to create models "
                    f"with explicit inputs. For Sequential models, you can "
                    f"call model(dummy_input) or model.build(input_shape) "
                    f"before using FeatureDistillation."
                )
            else:
                raise ValueError(
                    f"Could not create a feature extraction model for layer "
                    f"'{layer_name}'. This is likely because the model is a "
                    f"subclassed model that cannot be traversed using the "
                    f"standard layer API. Error: {e}"
                )

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
            loss = ops.mean(
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
            loss = 1.0 - similarity

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss

    def get_config(self):
        """Get configuration for serialization."""
        return {
            "loss_type": self.loss_type,
            "teacher_layer_name": self.teacher_layer_name,
            "student_layer_name": self.student_layer_name,
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        return cls(**config)


@keras_export("keras.distillation.MultiOutputDistillation")
class MultiOutputDistillation(BaseDistillationStrategy):
    """Multi-output distillation strategy for models with multiple outputs.

    Multi-output distillation handles models with multiple outputs, such as
    object detection models (classification + regression), multi-task learning
    models, or any model with multiple prediction heads. This strategy allows
    different distillation approaches for different outputs.

    How Multi-Output Distillation Works:

    1. Output Mapping: Map each output index to a specific distillation
       strategy. Different outputs can use different strategies based on their
       nature (classification vs regression, different loss functions, etc.).

    2. Strategy Application: Apply the appropriate strategy to each output
       pair (teacher output i → student output i).

    3. Loss Combination: Combine the losses from all outputs using
       configurable weights. This allows prioritizing certain outputs over
       others.

    When to Use Multi-Output Distillation:

    - Multi-Task Models: Models with multiple outputs (classification +
      regression)
    - Object Detection: Models with classification and bounding box outputs
    - Segmentation: Models with classification and mask outputs
    - Custom Architectures: Any model with multiple distinct outputs

    Output Strategy Selection:

    - Classification Outputs: Use `LogitsDistillation` with appropriate
      temperature
    - Regression Outputs: Use `LogitsDistillation` with lower temperature or
      `FeatureDistillation` with MSE loss
    - Feature Outputs: Use `FeatureDistillation` to transfer intermediate
      representations
    - Mixed Types: Combine different strategies for different outputs
    - Custom Losses: Each strategy can be subclassed to override
      `compute_loss` method

    Weight Configuration:

    - Equal Weights: Default behavior, all outputs weighted equally
    - Task-Specific Weights: Weight outputs based on task importance
    - Loss-Scale Weights: Adjust weights to balance different loss scales
    - Performance-Based: Weight outputs based on their impact on final
      performance

    Args:
        output_strategies: Dict mapping output indices to distillation
            strategies. Each strategy will be applied to the corresponding
            output. Example: `{0: LogitsDistillation(), 1:
            FeatureDistillation()}`
        weights: Dict mapping output indices to weights for combining losses.
            Defaults to equal weights for all outputs. Example:
            `{0: 1.0, 1: 0.5}`

    Example:

    ```python
    # Multi-output distillation for object detection
    strategy = MultiOutputDistillation(
        output_strategies={
            0: LogitsDistillation(temperature=3.0, output_index=0),
            1: LogitsDistillation(temperature=1.0, output_index=1)
        },
        weights={0: 1.0, 1: 0.5}  # Weight classification more heavily
    )

    # Custom multi-output strategy
    class CustomMultiOutputDistillation(MultiOutputDistillation):
        def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
            from keras import ops
            # Get the outputs to distill
            teacher_logits = teacher_outputs[0]
            student_logits = student_outputs[0]

            # Apply temperature scaling
            teacher_logits = teacher_logits / 3.0
            student_logits = student_logits / 3.0

            # Custom loss computation
            teacher_probs = ops.softmax(teacher_logits, axis=-1)
            student_probs = ops.softmax(student_logits, axis=-1)
            return ops.mean(
                keras.losses.kl_divergence(teacher_probs, student_probs)
            )

    class CustomFeatureDistillation(FeatureDistillation):
        def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
            from keras import ops
            teacher_features = teacher_outputs[0]
            student_features = student_outputs[0]
            return ops.mean(ops.abs(teacher_features - student_features))

    strategy = MultiOutputDistillation(
        output_strategies={
            0: CustomLogitsDistillation(temperature=4.0, output_index=0),
            1: CustomFeatureDistillation(output_index=1)
        }
    )

    # Equal weighting for all outputs
    strategy = MultiOutputDistillation(
        output_strategies={
            0: LogitsDistillation(temperature=3.0, output_index=0),
            1: LogitsDistillation(temperature=3.0, output_index=1),
            2: LogitsDistillation(temperature=3.0, output_index=2)
        }
        # weights=None (defaults to equal weights)
    )
    ```
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
        from keras.src.saving import serialization_lib

        return {
            "output_strategies": {
                k: serialization_lib.serialize_keras_object(v)
                for k, v in self.output_strategies.items()
            },
            "weights": self.weights,
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        from keras.src.saving import serialization_lib

        # JSON keys must be strings, so we convert them back to int
        config["output_strategies"] = {
            int(k): serialization_lib.deserialize_keras_object(v)
            for k, v in config["output_strategies"].items()
        }
        return cls(**config)
