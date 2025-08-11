"""Distiller class for knowledge distillation in KerasHub."""

import keras
from keras.src.api_export import keras_export


@keras_export("keras.distillation.Distiller")
class Distiller(keras.Model):
    """Knowledge distillation model that trains a student to mimic a teacher.
    This class implements knowledge distillation by training a smaller student
    model to replicate the behavior of a larger teacher model. The teacher model
    is kept frozen while the student learns from both ground truth labels and
    the teacher's soft predictions.
    Args:
        teacher: A keras.Model that provides target outputs. Must be frozen.
        student: A keras.Model that will be trained to mimic the teacher.
        strategies: List of distillation strategies to apply. Defaults to
            logits distillation only.
        student_loss_fn: Loss function for student's task loss. Defaults to
            SparseCategoricalCrossentropy.
        alpha: Weight for student loss vs distillation loss. Defaults to 0.5.
        temperature: Temperature for softening logits in distillation.
            Defaults to 2.0.
        **kwargs: Additional arguments passed to keras.Model.
    Example:
    ```python
    # Load teacher and student models
    teacher = keras.models.GemmaCausalLM.from_preset("gemma_2b_en")
    student = keras.models.GemmaCausalLM.from_preset("gemma_350m_en")
    # Freeze teacher
    teacher.trainable = False
    # Create distiller
    distiller = keras.distillation.Distiller(
        teacher=teacher,
        student=student,
        alpha=0.5,
        temperature=2.0
    )
    # Compile and train
    distiller.compile(optimizer=keras.optimizers.Adam())
    distiller.fit(X_train, y_train, epochs=3)
    # Use distilled student for inference
    trained_student = distiller.student
    output = trained_student.generate("Hello, world!")
    ```
    """

    def __init__(
        self,
        teacher,
        student,
        strategies=None,
        student_loss_fn=None,
        alpha=0.5,
        temperature=2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store teacher and student models
        self.teacher = teacher
        self.student = student

        # Ensure teacher is frozen
        self.teacher.trainable = False

        # Set up strategies
        if strategies is None:
            from keras.src.distillation.strategies import LogitsDistillation

            strategies = [LogitsDistillation(temperature=temperature)]
        self.strategies = strategies

        # Set up loss functions
        if student_loss_fn is None:
            # Use from_logits=False by default to handle both logits and
            # probabilities
            student_loss_fn = keras.losses.SparseCategoricalCrossentropy(
                from_logits=False
            )
        self.student_loss_fn = student_loss_fn
        self.alpha = alpha
        self.temperature = temperature

        # Track losses for monitoring
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = keras.metrics.Mean(
            name="distillation_loss"
        )
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    def call(self, inputs, training=None):
        """Forward pass - returns student outputs for inference."""
        return self.student(inputs, training=training)

    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        jit_compile=None,
        **kwargs,
    ):
        """Configure the distiller for training.
        Args:
            optimizer: Optimizer for training the student model.
            loss: Ignored - uses student_loss_fn and distillation strategies.
            metrics: Ignored - handled internally by distiller.
            **kwargs: Additional arguments passed to keras.Model.compile().
        """
        # Store compile arguments for use in train_step
        self._compile_optimizer = optimizer
        self._compile_metrics = metrics or []

        # Call parent compile with minimal configuration to avoid TrackedList
        # issues
        # We handle loss and metrics manually in train_step/test_step
        super().compile(
            optimizer=optimizer,
            loss=None,  # We handle loss manually in train_step
            metrics=None,  # We handle metrics manually to avoid TrackedList
            # issues
            **kwargs,
        )

    def reset_metrics(self):
        """Reset metrics to avoid TrackedList issues."""
        # Reset our custom loss trackers
        self.student_loss_tracker.reset_state()
        self.distillation_loss_tracker.reset_state()
        self.total_loss_tracker.reset_state()

    @property
    def metrics(self):
        """Return our custom metrics to avoid TrackedList issues."""
        return [
            self.student_loss_tracker,
            self.distillation_loss_tracker,
            self.total_loss_tracker,
        ]

    def train_step(self, data):
        """Custom training step for knowledge distillation."""
        x, y = data

        # Ensure y is the right shape for sparse categorical loss
        if hasattr(y, "shape") and len(y.shape) > 1 and y.shape[-1] == 1:
            y = keras.ops.squeeze(y, axis=-1)

        # Get teacher predictions (no gradients)
        teacher_outputs = self.teacher(x, training=False)
        teacher_outputs = keras.ops.stop_gradient(teacher_outputs)

        # Get student predictions
        student_outputs = self.student(x, training=True)

        # Compute student loss
        student_loss = self.student_loss_fn(y, student_outputs)

        # Compute distillation loss
        distillation_loss = 0.0
        for strategy in self.strategies:
            distillation_loss += strategy.compute_loss(
                teacher_outputs, student_outputs
            )

        # Combine losses
        total_loss = (
            self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        )

        # Add losses to model for Keras to handle gradients
        self.add_loss(total_loss)

        # Update loss trackers for monitoring
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.total_loss_tracker.update_state(total_loss)

        # Return metrics as simple dict (no TrackedList issues)
        return {
            "student_loss": self.student_loss_tracker.result(),
            "distillation_loss": self.distillation_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }

    def test_step(self, data):
        """Custom test step for knowledge distillation."""
        x, y = data

        # Ensure y is the right shape for sparse categorical loss
        if hasattr(y, "shape") and len(y.shape) > 1 and y.shape[-1] == 1:
            y = keras.ops.squeeze(y, axis=-1)

        # Get teacher predictions (no gradients)
        teacher_outputs = self.teacher(x, training=False)
        teacher_outputs = keras.ops.stop_gradient(teacher_outputs)

        # Get student predictions
        student_outputs = self.student(x, training=False)

        # Compute student loss
        student_loss = self.student_loss_fn(y, student_outputs)

        # Compute distillation loss
        distillation_loss = 0.0
        for strategy in self.strategies:
            distillation_loss += strategy.compute_loss(
                teacher_outputs, student_outputs
            )

        # Combine losses
        total_loss = (
            self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        )

        # Update loss trackers for monitoring
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.total_loss_tracker.update_state(total_loss)

        # Return metrics as simple dict (no TrackedList issues)
        return {
            "student_loss": self.student_loss_tracker.result(),
            "distillation_loss": self.distillation_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "teacher": self.teacher,
                "student": self.student,
                "strategies": self.strategies,
                "student_loss_fn": self.student_loss_fn,
                "alpha": self.alpha,
                "temperature": self.temperature,
            }
        )
        return config
