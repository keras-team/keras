import keras
from keras.src.api_export import keras_export
from keras.src.models.model import Model


@keras_export("keras.distillation.Distiller")
class Distiller(Model):
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
            alpha=1.0 means only student loss, alpha=0.0 means only
            distillation loss.
        temperature: Default temperature for distillation strategies that don't
            specify their own temperature. Used for softmax temperature scaling
            in knowledge distillation. Defaults to 3.0.
        name: Name of the distiller model.

    Examples:

    **Basic Knowledge Distillation:**

    ```python
    import keras
    import numpy as np
    from keras.distillation import Distiller, LogitsDistillation

    # Create teacher and student models
    teacher = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    student = keras.Sequential([
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Create distillation strategy (will use Distiller's default temperature)
    strategy = LogitsDistillation()

    # Create distiller with default temperature
    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategies=[strategy],
        alpha=0.7,  # 70% student loss, 30% distillation loss
        temperature=4.0  # Default temperature for all strategies
    )

    # Compile and train
    distiller.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy'
    )

    # Generate dummy data
    x_train = np.random.random((1000, 20))
    y_train = np.random.randint(0, 10, (1000,))

    # Train the distiller
    distiller.fit(x_train, y_train, epochs=10, batch_size=32)

    # Use the trained student model
    predictions = distiller.predict(x_train[:5])
    ```

    **Multi-Strategy Distillation:**

    ```python
    from keras.distillation import (
        Distiller, LogitsDistillation, FeatureDistillation
    )

    # Multiple distillation strategies
    strategies = [
        LogitsDistillation(),  # Will use Distiller's default temperature
        LogitsDistillation(temperature=2.0),  # Override with specific temp
        FeatureDistillation(
            loss_type="mse",
            teacher_layer_name="dense_1",
            student_layer_name="dense_1"
        )
    ]

    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategies=strategies,
        alpha=0.5,
        temperature=4.0  # Default temperature for strategies without one
    )
    ```

    **Multi-Output Model Distillation:**

    ```python
    from keras.distillation import MultiOutputDistillation

    # For models with multiple outputs
    multi_strategy = MultiOutputDistillation(
        output_strategies={
            0: LogitsDistillation(output_index=0),  # Uses default temperature
            1: LogitsDistillation(
                temperature=2.0, output_index=1
            )  # Override temperature
        },
        weights={0: 1.0, 1: 0.5}
    )

    distiller = Distiller(
        teacher=multi_output_teacher,
        student=multi_output_student,
        strategies=[multi_strategy],
        alpha=0.6,
        temperature=3.0  # Default temperature
    )
    ```

    **Custom Loss Function:**

    ```python
    # Using custom student loss function
    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategies=[LogitsDistillation()],  # Uses default temperature
        student_loss_fn=keras.losses.CategoricalCrossentropy(),
        alpha=0.8,
        temperature=5.0
    )
    ```
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
        **kwargs,
    ):
        # Extract input_mapping and output_mapping before super().__init__
        self.input_mapping = kwargs.pop("input_mapping", None)
        self.output_mapping = kwargs.pop("output_mapping", None)

        super().__init__(name=name, **kwargs)

        # Validate inputs
        self._validate_models(teacher, student)

        # Store configuration
        self.teacher = teacher
        self.student = student
        self.strategies = (
            strategies if isinstance(strategies, list) else [strategies]
        )
        self.alpha = alpha
        self.temperature = temperature

        # Apply default temperature to strategies that don't have one
        self._apply_default_temperature()

        # Set up student loss function
        if student_loss_fn is None:
            self.student_loss_fn = keras.losses.SparseCategoricalCrossentropy()
        else:
            self.student_loss_fn = student_loss_fn

        # Freeze teacher model
        self.teacher.trainable = False

        # Initialize loss tracking metrics
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = keras.metrics.Mean(
            name="distillation_loss"
        )
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    def _apply_default_temperature(self):
        """Apply default temperature to strategies that support it."""
        from keras.src.distillation.strategies import LogitsDistillation

        for strategy in self.strategies:
            if isinstance(strategy, LogitsDistillation):
                # Use the new method to set default temperature
                strategy.set_default_temperature(self.temperature)
            # Handle nested strategies in MultiOutputDistillation
            elif hasattr(strategy, "output_strategies"):
                for nested_strategy in strategy.output_strategies.values():
                    if isinstance(nested_strategy, LogitsDistillation):
                        nested_strategy.set_default_temperature(
                            self.temperature
                        )

    def _validate_models(self, teacher, student):
        """Validate that teacher and student are Keras models."""
        if not isinstance(teacher, keras.Model):
            raise ValueError(
                f"Teacher must be a keras.Model, got {type(teacher)}"
            )
        if not isinstance(student, keras.Model):
            raise ValueError(
                f"Student must be a keras.Model, got {type(student)}"
            )

    def call(self, inputs, training=None, **kwargs):
        """Forward pass returns student predictions."""
        return self.student(inputs, training=training, **kwargs)

    def _get_strategy_outputs(self, strategy, inputs, training=None):
        """Get the appropriate outputs for a specific strategy.

        For FeatureDistillation, this extracts intermediate features.
        For other strategies, this returns the final model outputs.
        """
        from keras.src.distillation.strategies import FeatureDistillation

        if isinstance(strategy, FeatureDistillation):
            # Extract features from specified intermediate layers
            teacher_features = strategy._get_teacher_features(
                self.teacher, inputs
            )
            student_features = strategy._get_student_features(
                self.student, inputs
            )
            return teacher_features, student_features
        else:
            # Use final model outputs for other strategies
            teacher_outputs = self.teacher(inputs, training=False)
            student_outputs = self.student(inputs, training=training)
            return teacher_outputs, student_outputs

    def _compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, training=None
    ):
        """Compute combined distillation loss.

        This method integrates distillation into Keras's standard training
        workflow.
        """
        # Get student predictions
        if y_pred is None:
            y_pred = self(x, training=training)

        # Normalize y_pred and y to lists for consistent handling
        if not isinstance(y_pred, (list, tuple)):
            y_pred = [y_pred]
        if y is not None and not isinstance(y, (list, tuple)):
            y = [y]

        # Compute student loss
        student_loss = 0.0
        if self.alpha > 0.0 and y is not None:
            # Try using compiled_loss first, fallback to student_loss_fn
            if (
                hasattr(self, "compiled_loss")
                and self.compiled_loss is not None
            ):
                student_loss = self.compiled_loss(
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                    regularization_losses=[],
                )
            else:
                # Fallback: use student_loss_fn directly
                if isinstance(y_pred, list) and len(y_pred) > 0:
                    # For multi-output, use first output for student loss
                    student_loss = self.student_loss_fn(y[0], y_pred[0])
                else:
                    student_loss = self.student_loss_fn(y, y_pred)

        # Compute distillation loss
        distillation_loss = 0.0
        if self.alpha < 1.0:
            for strategy in self.strategies:
                # Get appropriate outputs for this strategy
                teacher_outputs, student_outputs = self._get_strategy_outputs(
                    strategy, x, training=training
                )

                # Validate and compute loss for this strategy
                strategy.validate_outputs(teacher_outputs, student_outputs)
                strategy_loss = strategy.compute_loss(
                    teacher_outputs, student_outputs
                )
                distillation_loss += strategy_loss

        # Combine losses
        total_loss = (
            self.alpha * student_loss + (1.0 - self.alpha) * distillation_loss
        )

        # Update metrics
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.total_loss_tracker.update_state(total_loss)

        return total_loss

    def reset_metrics(self):
        """Reset all metrics."""
        super().reset_metrics()
        self.student_loss_tracker.reset_state()
        self.distillation_loss_tracker.reset_state()
        self.total_loss_tracker.reset_state()

    @property
    def metrics(self):
        """Return list of metrics."""
        return [
            self.total_loss_tracker,
            self.student_loss_tracker,
            self.distillation_loss_tracker,
        ]

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "temperature": self.temperature,
                "input_mapping": self.input_mapping,
                "output_mapping": self.output_mapping,
            }
        )
        return config
