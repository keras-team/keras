"""Distillation strategies for knowledge distillation."""

import keras
from keras import ops
from keras.src.api_export import keras_export


@keras_export("keras.distillation.BaseDistillationStrategy")
class BaseDistillationStrategy:
    """Base class for distillation strategies.
    Distillation strategies define how to compute the distillation loss
    between teacher and student outputs.
    To create custom distillation strategies, subclass this class and
    override the compute_loss method.
    """

    def compute_loss(self, teacher_outputs, student_outputs):
        """Compute distillation loss between teacher and student outputs.
        Args:
            teacher_outputs: Outputs from the teacher model.
            student_outputs: Outputs from the student model.
        Returns:
            Distillation loss tensor.
        """
        raise NotImplementedError("Subclasses must implement compute_loss")


@keras_export("keras.distillation.LogitsDistillation")
class LogitsDistillation(BaseDistillationStrategy):
    """Logits distillation with customizable loss functions.
    This strategy supports multiple loss functions for logits distillation,
    using Keras's built-in loss functions from the losses API.
    Args:
        temperature: Temperature for softening logits. Higher values
            make the distribution softer. Defaults to 2.0.
        loss_type: Type of loss function to use. Options:
            - "kl_divergence": KL divergence using keras.losses.kl_divergence
            - "mse": Mean squared error using keras.losses.mean_squared_error
            - "cross_entropy": Cross entropy using
              keras.losses.categorical_crossentropy
    """

    def __init__(self, temperature=2.0, loss_type="kl_divergence"):
        self.temperature = temperature
        self.loss_type = loss_type

        # Validate loss_type
        valid_loss_types = ["kl_divergence", "mse", "cross_entropy"]
        if loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}")

    def compute_loss(self, teacher_outputs, student_outputs):
        """Compute distillation loss using Keras built-in loss functions.
        Args:
            teacher_outputs: Logits from teacher model.
            student_outputs: Logits from student model.
        Returns:
            Distillation loss tensor.
        """
        # Apply temperature scaling
        teacher_logits = teacher_outputs / self.temperature
        student_logits = student_outputs / self.temperature

        if self.loss_type == "kl_divergence":
            # Convert to probabilities for KL divergence
            teacher_probs = ops.softmax(teacher_logits, axis=-1)
            student_probs = ops.softmax(student_logits, axis=-1)

            # Use Keras KLDivergence directly and reduce to scalar
            loss = ops.mean(
                keras.losses.kl_divergence(teacher_probs, student_probs)
            )

        elif self.loss_type == "mse":
            # Use Keras MeanSquaredError directly and reduce to scalar
            loss = ops.mean(
                keras.losses.mean_squared_error(teacher_logits, student_logits)
            )

        elif self.loss_type == "cross_entropy":
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
        }


@keras_export("keras.distillation.FeatureDistillation")
class FeatureDistillation(BaseDistillationStrategy):
    """Feature distillation strategy using Keras built-in loss functions.
    This strategy distills intermediate features from teacher to student,
    not just the final outputs.
    Args:
        loss_type: Type of loss function to use. Options:
            - "mse": Mean squared error using keras.losses.mean_squared_error
            - "cosine": Cosine similarity using keras.losses.cosine_similarity
    """

    def __init__(self, loss_type="mse"):
        self.loss_type = loss_type

        # Validate loss_type
        valid_loss_types = ["mse", "cosine"]
        if loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}")

    def compute_loss(self, teacher_features, student_features):
        """Compute feature distillation loss using Keras built-in loss
        functions.
        Args:
            teacher_features: Intermediate features from teacher model.
            student_features: Intermediate features from student model.
        Returns:
            Feature distillation loss tensor.
        """
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
        }
