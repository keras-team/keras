import keras
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.models.model import Model
from keras.src.saving import serialization_lib


@keras_export("keras.distillation.Distiller")
class Distiller(Model):
    """Distillation model for transferring knowledge from teacher to student.

    Knowledge distillation transfers knowledge from a large, complex model
    (`teacher`) to a smaller, simpler model (`student`). The student learns
    from both ground truth labels and the teacher's predictions, often
    achieving better performance than training on labels alone.

    How Knowledge Distillation Works:

    1. Teacher Model: A pre-trained, larger model that has learned complex
       patterns and relationships in the data. The teacher is frozen during
       distillation.

    2. Student Model: A smaller, simpler model that we want to train to mimic
       the teacher's behavior while being more efficient for deployment.

    3. Distillation Process: The student learns from two sources:
       - Hard targets: Traditional supervised learning with ground truth labels
       - Soft targets: The teacher's predictions, which contain information
         about class relationships and confidence levels

    4. Temperature Scaling: The teacher's logits are divided by a `temperature`
       parameter before applying softmax, creating "softer" probability
       distributions that are easier for the student to learn from.

    When to Use Knowledge Distillation:

    - Model Compression: Reduce model size for deployment on
      resource-constrained devices
    - Performance Improvement: Student models often outperform models trained
      only on labels
    - Transfer Learning: Leverage knowledge from large pre-trained models
    - Ensemble Distillation: Combine multiple teacher models into a single
      student

    Strategy Selection Guide:

    - `LogitsDistillation`: Most common approach. Transfers final output
      knowledge. Use for classification tasks where you want the student to
      learn the teacher's decision boundaries and confidence patterns.

    - `FeatureDistillation`: Transfers intermediate representations. Use when
      teacher and student have similar architectures, as it helps the student
      learn better internal representations. Often leads to better performance
      than logits-only.

    - `MultiOutputDistillation`: For models with multiple outputs (e.g.,
      object detection with classification and regression heads). Allows
      different distillation strategies for different outputs.

    - Custom Strategies: Create custom strategies by subclassing
      `BaseDistillationStrategy` and overriding the `compute_loss` method.

    Args:
        teacher: A trained `keras.Model` that serves as the knowledge source.
            The teacher model is frozen during distillation.
        student: A `keras.Model` to be trained through distillation. This model
            will learn from both ground truth labels and the teacher's
            predictions.
        strategy: Distillation strategy to apply. Can be `LogitsDistillation`,
            `FeatureDistillation`, `MultiOutputDistillation`, or a custom
            strategy.
        student_loss_weight: Weight for the student's supervised loss component.
            Must be between 0 and 1. Higher values emphasize ground truth
            labels, lower values emphasize teacher predictions. Defaults to 0.5.
        optimizer: Optimizer for training the student model. Can be a string
            identifier (e.g., `'adam'`) or an optimizer instance.
        student_loss: Loss function for the student's supervised learning
            component. Can be a string identifier or a loss function instance.
        metrics: List of metrics to track during training.
        name: Name for the distiller model. Defaults to `"distiller"`.
        **kwargs: Additional keyword arguments passed to the parent `Model`
            class.

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
            0: LogitsDistillation(temperature=3.0, output_index=0),
            1: LogitsDistillation(temperature=2.0, output_index=1)
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

        # Validate student_loss_weight
        if not isinstance(student_loss_weight, (int, float)):
            raise ValueError(
                f"student_loss_weight must be a number, got "
                f"{type(student_loss_weight)}"
            )
        if student_loss_weight < 0.0 or student_loss_weight > 1.0:
            raise ValueError(
                f"student_loss_weight must be between 0.0 and 1.0, "
                f"got {student_loss_weight}"
            )
        self.student_loss_weight = student_loss_weight

        # Validate metrics parameter
        if metrics is not None and not isinstance(metrics, (list, tuple)):
            raise ValueError(
                f"metrics must be a list or tuple, got {type(metrics)}"
            )

        # Convert string loss to function if needed
        if isinstance(student_loss, str):
            self._student_loss = keras.losses.get(student_loss)
            if self._student_loss is None:
                raise ValueError(
                    f"Unknown loss function: '{student_loss}'. "
                    "Please provide a valid loss function name or instance."
                )
        elif isinstance(student_loss, list):
            # Handle multi-output loss functions
            self._student_loss = []
            for i, loss in enumerate(student_loss):
                if isinstance(loss, str):
                    loss_fn = keras.losses.get(loss)
                    if loss_fn is None:
                        raise ValueError(
                            f"Unknown loss function at index {i}: '{loss}'. "
                            "Please provide valid loss function names or "
                            "instances."
                        )
                    self._student_loss.append(loss_fn)
                else:
                    self._student_loss.append(loss)
        else:
            self._student_loss = student_loss

        # Validate that we have a valid loss function
        if self._student_loss is None:
            raise ValueError(
                "Student loss function cannot be None. "
                "Please provide a valid 'student_loss' parameter."
            )

        # Validate architecture compatibility for feature distillation
        self._validate_architecture_compatibility(teacher, student)

        # Store strategy (single strategy only)
        if strategy is None:
            raise ValueError(
                "Distillation strategy cannot be None. "
                "Please provide a valid strategy such as LogitsDistillation, "
                "FeatureDistillation, or MultiOutputDistillation."
            )
        self.strategy = strategy

        # Validate strategy-specific compatibility
        self._validate_strategy_compatibility(teacher, student)

        # Freeze teacher model
        self.teacher.trainable = False

        # Initialize loss tracking metrics
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = keras.metrics.Mean(
            name="distillation_loss"
        )
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        # Compile the model with provided parameters
        self.compile(optimizer=optimizer, loss=student_loss, metrics=metrics)

    def _validate_models(self, teacher, student):
        """Validate that teacher and student models are compatible for
        distillation.

        This method performs comprehensive validation including:
        - Model type validation
        - Input shape compatibility
        - Output shape compatibility
        - Architecture compatibility for feature distillation
        - Data type compatibility
        """
        # Basic model type validation
        if not isinstance(teacher, keras.Model):
            raise ValueError(
                f"Teacher must be a keras.Model, got {type(teacher)}"
            )
        if not isinstance(student, keras.Model):
            raise ValueError(
                f"Student must be a keras.Model, got {type(student)}"
            )

        # Check if models are built
        # Subclassed models may not be built at this point and may not expose
        # symbolic `inputs`/`outputs`. We avoid hard failures here and rely on
        # runtime checks during the first call/fit. When symbolic tensors are
        # available, we perform full compatibility validation below.

        # Validate input compatibility
        self._validate_input_compatibility(teacher, student)

        # Validate output compatibility
        self._validate_output_compatibility(teacher, student)

        # Validate data type compatibility
        self._validate_dtype_compatibility(teacher, student)

    def _validate_input_compatibility(self, teacher, student):
        """Validate that teacher and student have compatible input shapes."""
        # If symbolic tensors are not available (subclassed models), skip.
        if not hasattr(teacher, "inputs") or not hasattr(student, "inputs"):
            return
        teacher_inputs = getattr(teacher, "inputs")
        student_inputs = getattr(student, "inputs")
        if teacher_inputs is None or student_inputs is None:
            return

        # Handle single input case
        if not isinstance(teacher_inputs, (list, tuple)):
            teacher_inputs = [teacher_inputs]
        if not isinstance(student_inputs, (list, tuple)):
            student_inputs = [student_inputs]

        # Check number of inputs
        if len(teacher_inputs) != len(student_inputs):
            raise ValueError(
                f"Teacher and student must have the same number of inputs. "
                f"Teacher has {len(teacher_inputs)} inputs, "
                f"student has {len(student_inputs)} inputs."
            )

        # Check input shapes
        for i, (teacher_input, student_input) in enumerate(
            zip(teacher_inputs, student_inputs)
        ):
            teacher_shape = teacher_input.shape
            student_shape = student_input.shape

            # Check if shapes are compatible (allowing for batch dimension
            # flexibility)
            if not self._shapes_are_compatible(teacher_shape, student_shape):
                raise ValueError(
                    f"Input {i} shapes are incompatible. "
                    f"Teacher input shape: {teacher_shape}, "
                    f"Student input shape: {student_shape}. "
                    f"All dimensions except batch size must match."
                )

    def _validate_output_compatibility(self, teacher, student):
        """Validate that teacher and student have compatible output shapes."""
        # If symbolic tensors are not available (subclassed models), skip.
        if not hasattr(teacher, "outputs") or not hasattr(student, "outputs"):
            return
        teacher_outputs = getattr(teacher, "outputs")
        student_outputs = getattr(student, "outputs")
        if teacher_outputs is None or student_outputs is None:
            return

        # Handle single output case
        if not isinstance(teacher_outputs, (list, tuple)):
            teacher_outputs = [teacher_outputs]
        if not isinstance(student_outputs, (list, tuple)):
            student_outputs = [student_outputs]

        # Check number of outputs
        if len(teacher_outputs) != len(student_outputs):
            raise ValueError(
                f"Teacher and student must have the same number of outputs. "
                f"Teacher has {len(teacher_outputs)} outputs, "
                f"student has {len(student_outputs)} outputs."
            )

        # Check output shapes
        for i, (teacher_output, student_output) in enumerate(
            zip(teacher_outputs, student_outputs)
        ):
            teacher_shape = teacher_output.shape
            student_shape = student_output.shape

            # For distillation, output shapes should be compatible
            if not self._shapes_are_compatible(teacher_shape, student_shape):
                raise ValueError(
                    f"Output {i} shapes are incompatible. "
                    f"Teacher output shape: {teacher_shape}, "
                    f"Student output shape: {student_shape}. "
                    f"All dimensions except batch size must match."
                )

    def _validate_dtype_compatibility(self, teacher, student):
        """Validate that teacher and student have compatible data types."""
        # If symbolic tensors are not available (subclassed models), skip.
        if not hasattr(teacher, "inputs") or not hasattr(student, "inputs"):
            return
        if teacher.inputs is None or student.inputs is None:
            return
        teacher_dtypes = [input.dtype for input in teacher.inputs]
        student_dtypes = [input.dtype for input in student.inputs]

        # Check input dtypes
        for i, (teacher_dtype, student_dtype) in enumerate(
            zip(teacher_dtypes, student_dtypes)
        ):
            if teacher_dtype != student_dtype:
                raise ValueError(
                    f"Input {i} data types are incompatible. "
                    f"Teacher dtype: {teacher_dtype}, "
                    f"Student dtype: {student_dtype}."
                )

        # Check output dtypes
        teacher_output_dtypes = [output.dtype for output in teacher.outputs]
        student_output_dtypes = [output.dtype for output in student.outputs]

        for i, (teacher_dtype, student_dtype) in enumerate(
            zip(teacher_output_dtypes, student_output_dtypes)
        ):
            if teacher_dtype != student_dtype:
                raise ValueError(
                    f"Output {i} data types are incompatible. "
                    f"Teacher output dtype: {teacher_dtype}, "
                    f"Student output dtype: {student_dtype}. "
                    f"Both models must use the same data type."
                )

    def _validate_architecture_compatibility(self, teacher, student):
        """Validate architecture compatibility for feature distillation."""
        # This validation is strategy-specific and will be called by strategies
        # that require specific architectural compatibility
        pass

    def _validate_strategy_compatibility(self, teacher, student):
        """Validate that the strategy is compatible with the teacher and student
        models."""
        if hasattr(self.strategy, "validate_model_compatibility"):
            self.strategy.validate_model_compatibility(teacher, student)

    def _shapes_are_compatible(self, shape1, shape2):
        """Check if two shapes are compatible (allowing for batch dimension
        flexibility)."""
        # Convert to lists for easier handling
        if hasattr(shape1, "as_list"):
            shape1 = shape1.as_list()
        elif hasattr(shape1, "__iter__"):
            shape1 = list(shape1)
        else:
            shape1 = [shape1]

        if hasattr(shape2, "as_list"):
            shape2 = shape2.as_list()
        elif hasattr(shape2, "__iter__"):
            shape2 = list(shape2)
        else:
            shape2 = [shape2]

        # Check if they have the same number of dimensions
        if len(shape1) != len(shape2):
            return False

        # Check all dimensions except the first (batch dimension)
        for dim1, dim2 in zip(shape1[1:], shape2[1:]):
            if dim1 is not None and dim2 is not None and dim1 != dim2:
                return False
        return True

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

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, training=None
    ):
        """Compute combined distillation loss.

        This method integrates distillation into Keras's standard training
        workflow. Users can override this method to implement custom
        distillation loss computation.

        Args:
            x: Input data.
            y: Target data.
            y_pred: Model predictions.
            sample_weight: Sample weights.
            training: Whether the model is in training mode.

        Returns:
            Combined loss tensor.

        Example:
            ```python
            # Custom distillation loss by overriding compute_loss
            class CustomDistiller(Distiller):
                def compute_loss(self, x=None, y=None, y_pred=None,
                               sample_weight=None, training=None):
                    # Custom student loss computation
                    student_loss = keras.losses.sparse_categorical_crossentropy(
                        y, y_pred
                    )

                    # Custom distillation loss computation
                    teacher_outputs = self.teacher(x, training=False)
                    student_outputs = self.student(x, training=training)

                    # Custom loss logic here
                    distillation_loss = self._custom_distillation_loss(
                        teacher_outputs, student_outputs
                    )

                    # Combine losses with custom weighting
                    total_loss = 0.7 * student_loss + 0.3 * distillation_loss

                    return total_loss

                def _custom_distillation_loss(self, teacher_outputs,
                                            student_outputs):
                    # Implement custom distillation loss logic
                    from keras import ops
                    return ops.mean(
                        ops.square(teacher_outputs - student_outputs)
                    )
            ```
        """
        # Normalize y_pred and y to lists for consistent handling
        if not isinstance(y_pred, (list, tuple)):
            y_pred = [y_pred]
        if y is not None and not isinstance(y, (list, tuple)):
            y = [y]

        # Compute student loss
        student_loss = 0.0
        if self.student_loss_weight > 0.0 and y is not None:
            # Use the configured loss function
            if (
                hasattr(self, "_student_loss")
                and self._student_loss is not None
            ):
                if isinstance(self._student_loss, list):
                    # Multi-output loss
                    if isinstance(y_pred, list) and len(y_pred) > 0:
                        # Validate lengths match
                        if len(y) != len(y_pred):
                            raise ValueError(
                                f"Number of targets ({len(y)}) must match "
                                f"number of predictions ({len(y_pred)}) for "
                                f"multi-output loss computation."
                            )
                        if len(self._student_loss) != len(y):
                            raise ValueError(
                                f"Number of loss functions "
                                f"({len(self._student_loss)}) must match "
                                f"number of outputs ({len(y)}) for "
                                f"multi-output loss computation."
                            )

                        # Compute loss for each output
                        student_loss = sum(
                            loss_fn(y[i], y_pred[i])
                            for i, loss_fn in enumerate(self._student_loss)
                        )
                    else:
                        # Single output with multi-output loss list
                        if len(self._student_loss) != 1:
                            raise ValueError(
                                f"Single output provided but "
                                f"{len(self._student_loss)} loss functions "
                                f"configured. Use a single loss function or "
                                f"provide multiple outputs."
                            )
                        student_loss = self._student_loss[0](y[0], y_pred[0])
                else:
                    # Single loss function
                    if isinstance(y_pred, list) and len(y_pred) > 0:
                        # Multi-output with single loss function
                        if len(y) != len(y_pred):
                            raise ValueError(
                                f"Number of targets ({len(y)}) must match "
                                f"number of predictions ({len(y_pred)}) for "
                                f"multi-output loss computation."
                            )
                        # Use first output for student loss (consistent
                        # behavior)
                        student_loss = self._student_loss(y[0], y_pred[0])
                    else:
                        # Single output with single loss function
                        student_loss = self._student_loss(y, y_pred)
            else:
                # No loss function configured - this is an error
                raise ValueError(
                    "Student loss function is not configured. "
                    "Please provide a valid 'student_loss' parameter to the "
                    "Distiller constructor. "
                    "Examples: 'sparse_categorical_crossentropy', "
                    "'categorical_crossentropy', or a custom loss function."
                )

            # Ensure student_loss is a scalar
            if hasattr(student_loss, "shape") and len(student_loss.shape) > 0:
                student_loss = ops.mean(student_loss)

        # Compute distillation loss
        distillation_loss = 0.0
        if self.student_loss_weight < 1.0:
            # Get teacher outputs
            teacher_outputs = self.teacher(x, training=False)

            # Apply the single strategy
            distillation_loss = self.strategy.compute_loss(
                teacher_outputs, y_pred
            )

            # Ensure distillation_loss is a scalar
            if (
                hasattr(distillation_loss, "shape")
                and len(distillation_loss.shape) > 0
            ):
                distillation_loss = ops.mean(distillation_loss)

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

        config = super().get_config()
        config.update(
            {
                "teacher": serialization_lib.serialize_keras_object(
                    self.teacher
                ),
                "student": serialization_lib.serialize_keras_object(
                    self.student
                ),
                "strategy": serialization_lib.serialize_keras_object(
                    self.strategy
                ),
                "student_loss_weight": self.student_loss_weight,
                "input_mapping": self.input_mapping,
                "output_mapping": self.output_mapping,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""

        config["teacher"] = serialization_lib.deserialize_keras_object(
            config["teacher"]
        )
        config["student"] = serialization_lib.deserialize_keras_object(
            config["student"]
        )
        config["strategy"] = serialization_lib.deserialize_keras_object(
            config["strategy"]
        )
        return cls(**config)
