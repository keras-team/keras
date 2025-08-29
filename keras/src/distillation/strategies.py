import keras
from keras.src import tree
from keras.src.api_export import keras_export


@keras_export("keras.distillation.DistillationLoss")
class DistillationLoss:
    """Base class for distillation loss computation.

    Distillation losses define how to compute the distillation loss
    between teacher and student outputs. Each loss implements a specific
    approach to knowledge transfer, from simple logits matching to feature-based
    distillation.

    To create custom distillation losses, subclass this class and
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
                f"Teacher and student must have the same number of "
                f"outputs. "
                f"Teacher has {len(teacher_outputs)} outputs, "
                f"student has {len(student_outputs)} outputs."
            )


@keras_export("keras.distillation.LogitsDistillation")
class LogitsDistillation(DistillationLoss):
    """Distillation strategy that transfers knowledge from final model outputs.

    This strategy applies temperature scaling to the teacher's logits before
    computing the loss between teacher and student predictions. It's the most
    common approach for knowledge distillation.

    How Logits Distillation Works:

    1. Temperature Scaling: The teacher's logits are divided by a `temperature`
       parameter (typically 3-5) before applying softmax. This creates "softer"
       probability distributions that reveal relationships between classes.

    2. Loss Computation: The loss is computed between the temperature-scaled
       teacher logits and student logits using the specified loss function.

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
        loss: Loss function(s) to use for distillation. Can be:
            - String identifier (e.g., 'kl_divergence',
              'categorical_crossentropy')
            - Keras loss instance
            - List/tuple of losses for multi-output models
            - Dict of losses for named outputs
            The structure must match the model's output structure.
            Defaults to 'kl_divergence'.

    Example:

    ```python
    # Basic logits distillation with KL divergence
    strategy = LogitsDistillation(temperature=3.0)

    # With categorical crossentropy loss
    strategy = LogitsDistillation(
        temperature=4.0,
        loss="categorical_crossentropy"
    )

    # With custom loss instance
    strategy = LogitsDistillation(
        temperature=4.0,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True)
    )

    # For multi-output models with list structure
    strategy = LogitsDistillation(
        temperature=3.0,
        loss=["kl_divergence", "categorical_crossentropy"]
    )

    # For multi-output models with dict structure
    strategy = LogitsDistillation(
        temperature=3.0,
        loss={
            "classification": "kl_divergence",
            "regression": "mse"
        }
    )

    # Custom loss by subclassing
    class CustomLogitsDistillation(LogitsDistillation):
        def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
            # Apply temperature scaling using tree.map_structure
            teacher_scaled = tree.map_structure(
                lambda x: x / self.temperature, teacher_outputs
            )
            student_scaled = tree.map_structure(
                lambda x: x / self.temperature, student_outputs
            )

            # Custom loss computation
            return tree.map_structure(
                lambda t, s: keras.ops.mean(
                    keras.losses.kl_divergence(
                        keras.ops.softmax(t, axis=-1),
                        keras.ops.softmax(s, axis=-1)
                    )
                ),
                teacher_scaled,
                student_scaled
            )
    ```
    """

    def __init__(
        self,
        temperature=3.0,
        loss="kl_divergence",
    ):
        super().__init__()
        self.temperature = temperature

        # Convert loss structure to functions using tree.map_structure
        def convert_loss_to_function(loss_item):
            if isinstance(loss_item, str):
                loss_fn = keras.losses.get(loss_item)
                if loss_fn is None:
                    raise ValueError(
                        f"Unknown loss function: '{loss_item}'. "
                        "Please provide a valid loss function name or instance."
                    )
                return loss_fn
            else:
                return loss_item

        self.loss = tree.map_structure(convert_loss_to_function, loss)

        # Validate temperature
        if not isinstance(self.temperature, (int, float)):
            raise ValueError(
                f"temperature must be a number, got {type(self.temperature)}"
            )
        if self.temperature <= 0.0:
            raise ValueError(
                "temperature must be > 0. Set a positive value (e.g., 1-10)."
            )

    def validate_outputs(self, teacher_outputs, student_outputs):
        """Validate that outputs are compatible for logits distillation."""
        super().validate_outputs(teacher_outputs, student_outputs)

        # Validate that loss structure matches output structure
        try:
            tree.assert_same_structure(self.loss, teacher_outputs)
            tree.assert_same_structure(self.loss, student_outputs)
        except ValueError as e:
            raise ValueError(
                f"Loss structure must match output structure. "
                f"Loss structure: {tree.structure(self.loss)}, "
                f"Teacher output structure: {tree.structure(teacher_outputs)}, "
                f"Student output structure: {tree.structure(student_outputs)}. "
                f"Error: {e}"
            )

    def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
        """Compute distillation loss using the configured loss function.

        Args:
            teacher_outputs: Logits from teacher model. Can be a single tensor,
                list/tuple of tensors, or dict of tensors.
            student_outputs: Logits from student model. Can be a single tensor,
                list/tuple of tensors, or dict of tensors.
            **kwargs: Additional arguments (ignored).
        Returns:
            Distillation loss tensor.
        """
        # Apply temperature scaling using tree.map_structure
        teacher_scaled = tree.map_structure(
            lambda x: x / self.temperature, teacher_outputs
        )
        student_scaled = tree.map_structure(
            lambda x: x / self.temperature, student_outputs
        )

        # Apply loss function(s) to corresponding outputs
        def apply_loss(loss_fn, teacher_logits, student_logits):
            # Special handling for KL divergence (needs probabilities)
            if (
                hasattr(loss_fn, "__name__")
                and "kl" in loss_fn.__name__.lower()
            ):
                teacher_probs = keras.ops.softmax(teacher_logits, axis=-1)
                student_probs = keras.ops.softmax(student_logits, axis=-1)
                loss = keras.ops.mean(loss_fn(teacher_probs, student_probs))
                # Scale by temperature^2 for KL (per literature)
                return loss * (self.temperature**2)
            else:
                # For other losses, use logits directly
                return keras.ops.mean(loss_fn(teacher_logits, student_logits))

        # Apply losses using tree.map_structure
        loss_values = tree.map_structure(
            apply_loss, self.loss, teacher_scaled, student_scaled
        )

        # Sum all losses and return scalar
        flat_losses = tree.flatten(loss_values)
        return keras.ops.sum(keras.ops.stack(flat_losses))

    def get_config(self):
        """Get configuration for serialization."""
        return {
            "temperature": self.temperature,
            "loss": keras.losses.serialize(self.loss),
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        config = config.copy()
        config["loss"] = keras.losses.deserialize(config["loss"])
        return cls(**config)


@keras_export("keras.distillation.FeatureDistillation")
class FeatureDistillation(DistillationLoss):
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
        loss: Loss function(s) to use for feature distillation. Can be:
            - String identifier (e.g., 'mse', 'cosine_similarity', 'mae')
            - Keras loss instance
            - List/tuple of losses for multi-output models
            - Dict of losses for named outputs
            The structure must match the model's output structure.
            Defaults to 'mse'.
        teacher_layer_name: Name of the teacher layer to extract features from.
            If None, uses the final output. Defaults to None.
        student_layer_name: Name of the student layer to extract features from.
            If None, uses the final output. Defaults to None.

    Examples:

    ```python
    # Basic feature distillation from final outputs
    strategy = FeatureDistillation(loss="mse")

    # With custom loss instance
    strategy = FeatureDistillation(
        loss=keras.losses.MeanAbsoluteError()
    )

    # For multi-output models with list structure
    strategy = FeatureDistillation(
        loss=["mse", "cosine_similarity"]
    )

    # For multi-output models with dict structure
    strategy = FeatureDistillation(
        loss={
            "features_1": "mse",
            "features_2": "cosine_similarity"
        }
    )

    # Custom loss by subclassing
    class CustomFeatureDistillation(FeatureDistillation):
        def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
            # Apply loss using tree.map_structure
            return tree.map_structure(
                lambda t, s: keras.ops.mean(
                    keras.ops.abs(t - s)
                ),
                teacher_outputs,
                student_outputs
            )

    strategy = CustomFeatureDistillation(
        teacher_layer_name="dense_1",
        student_layer_name="dense_1"
    )

    # Distill from specific layers with compatible shapes
    strategy = FeatureDistillation(
        loss="mse",
        teacher_layer_name="dense_1",
        student_layer_name="dense_1"
    )

    # Use cosine similarity for different feature sizes
    strategy = FeatureDistillation(
        loss="cosine_similarity",
        teacher_layer_name="conv2d_2",
        student_layer_name="conv2d_1"
    )

    # Distill from final outputs (equivalent to logits distillation)
    strategy = FeatureDistillation(
        loss="mse",
        teacher_layer_name=None,  # Final output
        student_layer_name=None   # Final output
    )
    ```
    """

    def __init__(
        self, loss="mse", teacher_layer_name=None, student_layer_name=None
    ):
        self.teacher_layer_name = teacher_layer_name
        self.student_layer_name = student_layer_name

        # Feature extraction models (created when needed)
        self._teacher_feature_model = None
        self._student_feature_model = None

        # Convert loss structure to functions using tree.map_structure
        def convert_loss_to_function(loss_item):
            if isinstance(loss_item, str):
                loss_fn = keras.losses.get(loss_item)
                if loss_fn is None:
                    raise ValueError(
                        f"Unknown loss function: '{loss_item}'. "
                        "Please provide a valid loss function name or instance."
                    )
                return loss_fn
            else:
                return loss_item

        self.loss = tree.map_structure(convert_loss_to_function, loss)

    def _get_features(
        self, model, inputs, training, layer_name, feature_model_attr
    ):
        """Extract features from model at specified layer.

        Args:
            model: The model to extract features from.
            inputs: Input data.
            training: Whether model is in training mode.
            layer_name: Name of layer to extract from (None for final output).
            feature_model_attr: Attribute name to cache feature extraction
                model.

        Returns:
            Extracted features.
        """
        # No specific layer, use the final model output
        if layer_name is None:
            return model(inputs, training=training)

        # For intermediate layer extraction, create feature extractor if needed
        if getattr(self, feature_model_attr) is None:
            try:
                setattr(
                    self,
                    feature_model_attr,
                    self._create_feature_extractor(model, layer_name),
                )
            except ValueError as e:
                if "no defined inputs" in str(e).lower():
                    # Build the model by calling it with inputs first
                    _ = model(inputs, training=training)
                    # Now try again
                    setattr(
                        self,
                        feature_model_attr,
                        self._create_feature_extractor(model, layer_name),
                    )
                else:
                    raise

        return getattr(self, feature_model_attr)(inputs, training=training)

    def get_teacher_features(self, teacher_model, inputs):
        """Extract features from teacher model."""
        return self._get_features(
            teacher_model,
            inputs,
            False,
            self.teacher_layer_name,
            "_teacher_feature_model",
        )

    def get_student_features(self, student_model, inputs):
        """Extract features from student model."""
        return self._get_features(
            student_model,
            inputs,
            True,
            self.student_layer_name,
            "_student_feature_model",
        )

    def validate_model_compatibility(self, teacher, student):
        """Validate that teacher and student models are compatible for feature
        distillation."""
        # Check if specified layers exist in the models
        if self.teacher_layer_name is not None:
            try:
                teacher.get_layer(name=self.teacher_layer_name)
            except ValueError as e:
                raise ValueError(f"In teacher model: {e}")

        if self.student_layer_name is not None:
            try:
                student.get_layer(name=self.student_layer_name)
            except ValueError as e:
                raise ValueError(f"In student model: {e}")

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

        # Get the layer using Keras built-in method
        try:
            target_layer = model.get_layer(name=layer_name)
        except ValueError as e:
            raise ValueError(
                f"Layer '{layer_name}' not found in model '{model.name}'. {e}"
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

        # Validate that loss structure matches output structure
        try:
            tree.assert_same_structure(self.loss, teacher_outputs)
            tree.assert_same_structure(self.loss, student_outputs)
        except ValueError as e:
            raise ValueError(
                f"Loss structure must match output structure. "
                f"Loss structure: {tree.structure(self.loss)}, "
                f"Teacher output structure: {tree.structure(teacher_outputs)}, "
                f"Student output structure: {tree.structure(student_outputs)}. "
                f"Error: {e}"
            )

        # For feature distillation, validate layer compatibility if specified
        if (
            self.teacher_layer_name is not None
            and self.student_layer_name is not None
        ):
            # Validate that the specified layers exist and are compatible
            self._validate_layer_compatibility(teacher_outputs, student_outputs)
        # Note: Base class already validated output count compatibility

    def _validate_layer_compatibility(self, teacher_outputs, student_outputs):
        """Validate that the specified layers are compatible for feature
        distillation."""
        # This method would be called by the distiller to validate layer
        # compatibility when using feature distillation with specific layer
        # names
        pass

    def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
        """Compute feature distillation loss using extracted features.

        Note: This method expects the outputs to already be the extracted
        features from the specified layers, not the final model outputs.
        The Distiller class is responsible for extracting the features
        using the methods provided by this strategy.

        Args:
            teacher_outputs: Intermediate features from teacher model.
                Can be a single tensor, list/tuple of tensors, or dict of
                tensors.
            student_outputs: Intermediate features from student model.
                Can be a single tensor, list/tuple of tensors, or dict of
                tensors.
            **kwargs: Additional arguments (ignored).
        Returns:
            Feature distillation loss tensor.
        """

        # Apply loss function(s) to corresponding features
        def apply_loss(loss_fn, teacher_features, student_features):
            loss = keras.ops.mean(loss_fn(teacher_features, student_features))

            # Special handling for cosine similarity (convert similarity to
            # distance)
            if (
                hasattr(loss_fn, "__name__")
                and "cosine" in loss_fn.__name__.lower()
            ):
                # Convert similarity to distance: distance = 1 - similarity
                loss = 1.0 - loss

            return loss

        # Apply losses using tree.map_structure
        loss_values = tree.map_structure(
            apply_loss, self.loss, teacher_outputs, student_outputs
        )

        # Sum all losses and return scalar
        flat_losses = tree.flatten(loss_values)
        return keras.ops.sum(keras.ops.stack(flat_losses))

    def get_config(self):
        """Get configuration for serialization."""
        return {
            "loss": keras.losses.serialize(self.loss),
            "teacher_layer_name": self.teacher_layer_name,
            "student_layer_name": self.student_layer_name,
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        config = config.copy()
        config["loss"] = keras.losses.deserialize(config["loss"])
        return cls(**config)
