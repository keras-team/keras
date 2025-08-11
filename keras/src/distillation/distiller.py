"""Knowledge Distillation implementation for Keras.

This module provides a Distiller class that enables knowledge distillation
between teacher and student models using various distillation strategies.
"""

import keras
from keras import ops
from keras.src.api_export import keras_export


@keras_export("keras.distillation.Distiller")
class Distiller(keras.Model):
    """Knowledge Distillation model.

    This class implements knowledge distillation by combining a teacher model
    and a student model with configurable distillation strategies.

    The Distiller integrates seamlessly with Keras's training infrastructure
    by overriding the _compute_loss method, allowing standard model.fit(),
    model.evaluate(), and model.predict() workflows to work correctly.

    Args:
        teacher: The teacher model (will be frozen during training).
        student: The student model to be trained.
        strategies: List of distillation strategies to apply.
        student_loss_fn: Loss function for student predictions. Defaults to
            sparse categorical crossentropy.
        alpha: Weight for combining student loss and distillation loss.
            alpha=1.0 means only student loss, alpha=0.0 means only distillation loss.
        temperature: Temperature for softmax in distillation (used by strategies).
        name: Name of the distiller model.
    """

    def __init__(
        self,
        teacher,
        student,
        strategies,
        student_loss_fn=None,
        alpha=0.5,
        temperature=3.0,
        name="distiller",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        # Validate inputs
        self._validate_models(teacher, student)
        
        # Store configuration
        self.teacher = teacher
        self.student = student
        self.strategies = strategies if isinstance(strategies, list) else [strategies]
        self.alpha = alpha
        self.temperature = temperature
        
        # Set up student loss function
        if student_loss_fn is None:
            self.student_loss_fn = keras.losses.SparseCategoricalCrossentropy()
        else:
            self.student_loss_fn = student_loss_fn
            
        # Freeze teacher model
        self.teacher.trainable = False
        
        # Initialize loss tracking metrics
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = keras.metrics.Mean(name="distillation_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    def _validate_models(self, teacher, student):
        """Validate that teacher and student are Keras models."""
        if not isinstance(teacher, keras.Model):
            raise ValueError(f"Teacher must be a keras.Model, got {type(teacher)}")
        if not isinstance(student, keras.Model):
            raise ValueError(f"Student must be a keras.Model, got {type(student)}")

    def call(self, inputs, training=None, **kwargs):
        """Forward pass returns student predictions."""
        return self.student(inputs, training=training, **kwargs)

    def _compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=None):
        """Compute combined distillation loss.
        
        This method integrates distillation into Keras's standard training workflow.
        """
        # Get student predictions
        if y_pred is None:
            y_pred = self(x, training=training)
        
        # Get teacher predictions (no gradients)
        teacher_outputs = self.teacher(x, training=False)
        teacher_outputs = keras.ops.stop_gradient(teacher_outputs)
        
        # Normalize outputs for consistent handling
        student_outputs = [y_pred] if not isinstance(y_pred, (list, tuple)) else list(y_pred)
        teacher_outputs = [teacher_outputs] if not isinstance(teacher_outputs, (list, tuple)) else list(teacher_outputs)
        
        # Validate outputs with strategies
        for strategy in self.strategies:
            if hasattr(strategy, 'validate_outputs'):
                strategy.validate_outputs(teacher_outputs, student_outputs)
        
        # Compute student loss (supervised learning)
        if y is not None:
            student_loss = self.student_loss_fn(y, student_outputs[0])
        else:
            student_loss = 0.0
            
        # Compute distillation loss
        distillation_loss = 0.0
        for strategy in self.strategies:
            distillation_loss += strategy.compute_loss(teacher_outputs, student_outputs)
            
        # Combine losses
        total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        # Update metrics
        self.student_loss_tracker.update_state(student_loss if self.alpha > 0 else 0.0)
        self.distillation_loss_tracker.update_state(distillation_loss if self.alpha < 1 else 0.0)
        self.total_loss_tracker.update_state(total_loss)
        
        return total_loss

    @property
    def metrics(self):
        """Return metrics for monitoring."""
        # Combine parent metrics with our loss trackers
        parent_metrics = []
        if hasattr(super(), 'metrics'):
            for metric in super().metrics:
                if hasattr(metric, 'variables') and hasattr(metric, 'update_state'):
                    parent_metrics.append(metric)
        
        return parent_metrics + [
            self.student_loss_tracker,
            self.distillation_loss_tracker,
            self.total_loss_tracker,
        ]

    def reset_metrics(self):
        """Reset all metrics."""
        try:
            super().reset_metrics()
        except AttributeError:
            pass
        
        self.student_loss_tracker.reset_state()
        self.distillation_loss_tracker.reset_state()
        self.total_loss_tracker.reset_state()

    def get_config(self):
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "teacher": keras.utils.serialize_keras_object(self.teacher),
            "student": keras.utils.serialize_keras_object(self.student),
            "strategies": [keras.utils.serialize_keras_object(s) for s in self.strategies],
            "student_loss_fn": keras.utils.serialize_keras_object(self.student_loss_fn),
            "alpha": self.alpha,
            "temperature": self.temperature,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        config = config.copy()
        config["teacher"] = keras.utils.deserialize_keras_object(config["teacher"])
        config["student"] = keras.utils.deserialize_keras_object(config["student"])
        config["strategies"] = [
            keras.utils.deserialize_keras_object(s) for s in config["strategies"]
        ]
        config["student_loss_fn"] = keras.utils.deserialize_keras_object(
            config["student_loss_fn"]
        )
        return cls(**config) 