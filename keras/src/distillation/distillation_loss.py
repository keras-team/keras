import keras
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.saving import serialization_lib


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
        # keras.tree.assert_same_structure validates that structures match,
        # including the number of outputs, so no additional checks needed
        keras.tree.assert_same_structure(teacher_outputs, student_outputs)

    def validate_model_compatibility(self, teacher, student):
        """Validate that teacher and student models are compatible.

        This method ensures that the teacher and student models are compatible
        for the specific distillation strategy. It should check model structure,
        layer availability, and other strategy-specific requirements.

        Args:
            teacher: The teacher model.
            student: The student model.
        Raises:
            ValueError: If models are not compatible with this strategy.
        """
        # can be overridden by subclasses
        pass


@keras_export("keras.distillation.FeatureDistillation")
class FeatureDistillation(DistillationLoss):
    """Feature distillation strategy using intermediate layer representations.

    Feature distillation transfers knowledge from intermediate layers of the
    teacher model to corresponding layers of the student model. This approach
    helps the student learn better internal representations and often leads
    to better performance compared to logits-only distillation.

    Args:
        loss: Loss function to use for feature distillation. Can be:
            - String identifier (e.g., 'mse', 'cosine_similarity', 'mae')
            - Keras loss instance
            - Nested structure of losses matching the layer output structure
            Defaults to 'mse'.
        teacher_layer_name: Name of the teacher layer to extract features from.
            If None, uses the final output. Defaults to None.
        student_layer_name: Name of the student layer to extract features from.
            If None, uses the final output. Defaults to None.

    Examples:

    ```python
    # Basic feature distillation from final outputs
    strategy = FeatureDistillation(loss="mse")

    # Distill from specific intermediate layers
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

    # With custom loss instance
    strategy = FeatureDistillation(
        loss=keras.losses.MeanAbsoluteError()
    )

    # For multi-output models
    strategy = FeatureDistillation(
        loss=["mse", "cosine_similarity"]
    )
    ```
    """

    def __init__(
        self, loss="mse", teacher_layer_name=None, student_layer_name=None
    ):
        self.teacher_layer_name = teacher_layer_name
        self.student_layer_name = student_layer_name

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

    def _validate_layer_compatibility(self, teacher_outputs, student_outputs):
        """Validate that the specified layers are compatible for feature
        distillation."""
        # This method would be called by the distiller to validate layer
        # compatibility when using feature distillation with specific layer
        # names
        pass

    def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
        """Compute feature distillation loss using extracted features.

        Args:
            teacher_outputs: Extracted features from the specified teacher
                layer.
            student_outputs: Extracted features from the specified student
                layer.
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


@keras_export("keras.distillation.LogitsDistillation")
class LogitsDistillation(DistillationLoss):
    """Distillation strategy that transfers knowledge from final model outputs.

    This strategy applies temperature scaling to the teacher's logits before
    computing the loss between teacher and student predictions. It's the most
    common approach for knowledge distillation.

    Args:
        temperature: Temperature for softmax scaling. Higher values produce
            softer probability distributions that are easier for the student to
            learn. Typical values range from 3-5. Defaults to 3.0.
        loss: Loss function to use for distillation. Can be:
            - String identifier (e.g., 'kl_divergence',
              'categorical_crossentropy')
            - Keras loss instance
            - Nested structure of losses matching the model output structure
            Defaults to 'kl_divergence'.

    Examples:

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

    # For multi-output models
    strategy = LogitsDistillation(
        temperature=3.0,
        loss=["kl_divergence", "categorical_crossentropy"]
    )
    ```
    """

    def __init__(
        self,
        temperature=3.0,
        loss="kl_divergence",
    ):
        self.temperature = temperature

        # Convert loss structure to functions using tree.map_structure
        def convert_loss_to_function(loss_item):
            if isinstance(loss_item, str):
                loss_fn = keras.losses.get(loss_item)
                if loss_fn is None:
                    raise ValueError(f"Unknown loss function: {loss_item}")
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
            lambda x: keras.ops.divide(x, self.temperature), teacher_outputs
        )
        student_scaled = tree.map_structure(
            lambda x: keras.ops.divide(x, self.temperature), student_outputs
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
            "loss": serialization_lib.serialize_keras_object(self.loss),
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        config = config.copy()
        config["loss"] = keras.losses.deserialize(config["loss"])
        return cls(**config)
