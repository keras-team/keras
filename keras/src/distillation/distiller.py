import keras
from keras.src.api_export import keras_export
from keras.src.models.model import Model


@keras_export("keras.distillation.Distiller")
class Distiller(Model):
    """Knowledge Distillation model for transferring knowledge from teacher to student.

    Knowledge distillation transfers knowledge from a large, complex model (teacher)
    to a smaller, simpler model (student). The student learns from both ground truth
    labels and the teacher's predictions, often achieving better performance than
    training on labels alone.

    How Knowledge Distillation Works:

    1. Teacher Model: A pre-trained, larger model that has learned complex patterns
       and relationships in the data. The teacher is frozen during distillation.

    2. Student Model: A smaller, simpler model that we want to train to mimic
       the teacher's behavior while being more efficient for deployment.

    3. Distillation Process: The student learns from two sources:
       - Hard targets: Traditional supervised learning with ground truth labels
       - Soft targets: The teacher's predictions, which contain rich information
         about class relationships and confidence levels

    4. Temperature Scaling: The teacher's logits are divided by a temperature
       parameter before applying softmax, creating "softer" probability distributions
       that are easier for the student to learn from.

    When to Use Knowledge Distillation:

    - Model Compression: Reduce model size for deployment on resource-constrained devices
    - Performance Improvement: Student models often outperform models trained only on labels
    - Transfer Learning: Leverage knowledge from large pre-trained models
    - Ensemble Distillation: Combine multiple teacher models into a single student

    Strategy Selection Guide:

    - LogitsDistillation: Most common approach. Transfers final output knowledge.
      Best for classification tasks where you want the student to learn the teacher's
      decision boundaries and confidence patterns.

    - FeatureDistillation: Transfers intermediate representations. Best when teacher
      and student have similar architectures, as it helps the student learn better
      internal representations. Often leads to better performance than logits-only.

    - MultiOutputDistillation: For complex models with multiple outputs (e.g.,
      object detection with classification and regression heads). Allows different
      distillation strategies for different outputs.

    Args:
        teacher: A trained keras.Model that serves as the knowledge source.
            The teacher model is frozen during distillation.
        student: A keras.Model to be trained through distillation. This model
            will learn from both ground truth labels and the teacher's predictions.
        strategy: Distillation strategy or list of strategies. Can be a single
            strategy (e.g., LogitsDistillation) or a list of strategies for
            multi-strategy distillation.
        student_loss_weight: Weight for the student's supervised loss component.
            Must be between 0 and 1. Higher values emphasize ground truth labels,
            lower values emphasize teacher predictions. Defaults to 0.5.
        optimizer: Optimizer for training the student model. Can be a string
            identifier (e.g., 'adam') or an optimizer instance.
        student_loss: Loss function for the student's supervised learning component.
            Can be a string identifier or a loss function instance.
        metrics: List of metrics to track during training.
        name: Name for the distiller model. Defaults to "distiller".
        **kwargs: Additional keyword arguments passed to the parent Model class.

    Example:

    ```python
    # Load pre-trained teacher model from KerasHub
    import keras_hub as hub
    
    teacher = hub.models.CausalLM.from_preset("gemma3_4b_en")
    student = hub.models.CausalLM.from_preset("gemma2_2b_en")
    
    # Create distillation strategy
    strategy = LogitsDistillation(temperature=3.0)

    # Create distiller
    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategy=strategy,
        student_loss_weight=0.7,
        optimizer='adam',
        student_loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the distiller
    distiller.fit(x_train, y_train, epochs=10, validation_split=0.2)

    # Get the trained student model
    trained_student = distiller.get_student_model()
    ```

    For multi-output models:

    ```python
    # Create multi-output strategy
    multi_strategy = MultiOutputDistillation(
        output_strategies={
            0: LogitsDistillation(temperature=3.0, output_index=0),  # Classification
            1: LogitsDistillation(temperature=2.0, output_index=1)   # Regression
        },
        weights={0: 1.0, 1: 0.5}  # Weight classification more heavily
    )

    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategy=multi_strategy,
        student_loss_weight=0.5,
        optimizer='adam',
        student_loss=['sparse_categorical_crossentropy', 'mse']
    )
    ```
    """

    def __init__(
        self,
        teacher,
        student,
        strategy,
        student_loss_weight=0.5,
        optimizer="adam",
        student_loss="sparse_categorical_crossentropy",
        metrics=None,
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
        self.student_loss_weight = student_loss_weight

        # Handle strategy input - can be single strategy or list
        if isinstance(strategy, list):
            self.strategies = strategy
        else:
            self.strategies = [strategy]

        # Freeze teacher model
        self.teacher.trainable = False

        # Initialize loss tracking metrics
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = keras.metrics.Mean(
            name="distillation_loss"
        )
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        # Compile the model with provided parameters
        self.compile(
            optimizer=optimizer, loss=student_loss, metrics=metrics or []
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

    def get_student_model(self):
        """Get the trained student model for independent use.

        This method returns the student model that has been trained through
        the distillation process. The returned model can be used independently
        for inference, further training, or saving.

        Returns:
            keras.Model: The trained student model.

        Example:
            ```python
            # After training the distiller
            distiller.fit(x_train, y_train, epochs=10)

            # Get the trained student model
            trained_student = distiller.get_student_model()

            # Use the student model independently
            predictions = trained_student.predict(x_test)

            # Save the student model
            trained_student.save('my_student_model.keras')

            # Further train the student model
            trained_student.compile(
                optimizer='adam', loss='sparse_categorical_crossentropy'
            )
            trained_student.fit(x_new, y_new, epochs=5)
            ```
        """
        return self.student

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
        if self.student_loss_weight > 0.0 and y is not None:
            # Try using compiled_loss first, fallback to default loss
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
                # Fallback: use default loss function
                if isinstance(y_pred, list) and len(y_pred) > 0:
                    # For multi-output, use first output for student loss
                    student_loss = keras.losses.sparse_categorical_crossentropy(
                        y[0], y_pred[0]
                    )
                else:
                    student_loss = keras.losses.sparse_categorical_crossentropy(
                        y, y_pred
                    )

        # Compute distillation loss
        distillation_loss = 0.0
        if self.student_loss_weight < 1.0:
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
            self.student_loss_weight * student_loss
            + (1.0 - self.student_loss_weight) * distillation_loss
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
        from keras.src.saving import serialization_lib

        config = super().get_config()
        config.update(
            {
                "teacher": serialization_lib.serialize_keras_object(
                    self.teacher
                ),
                "student": serialization_lib.serialize_keras_object(
                    self.student
                ),
                "strategy": [
                    serialization_lib.serialize_keras_object(s)
                    for s in self.strategies
                ],
                "student_loss_weight": self.student_loss_weight,
                "input_mapping": self.input_mapping,
                "output_mapping": self.output_mapping,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        from keras.src.saving import serialization_lib

        config["teacher"] = serialization_lib.deserialize_keras_object(
            config["teacher"]
        )
        config["student"] = serialization_lib.deserialize_keras_object(
            config["student"]
        )
        config["strategy"] = [
            serialization_lib.deserialize_keras_object(s)
            for s in config["strategy"]
        ]
        return cls(**config)
