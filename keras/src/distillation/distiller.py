import keras
from keras.src import tree
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

    - Multiple Strategies: For models with multiple outputs (e.g., object
      detection with classification and regression heads), pass a list of
      strategies with corresponding weights. Each strategy will be applied to
      its corresponding output.

    - Custom Strategies: Create custom strategies by subclassing
      `DistillationLoss` and overriding the `compute_loss` method.

    Args:
        teacher: A trained `keras.Model` that serves as the knowledge source.
            The teacher model is frozen during distillation.
        student: A `keras.Model` to be trained through distillation. This model
            will learn from both ground truth labels and the teacher's
            predictions.
        strategy: Single distillation strategy to apply. Can be
            `LogitsDistillation`, `FeatureDistillation`, or a custom strategy.
            Use `strategies` for multiple strategies.
        strategies: List of distillation strategies to apply. Each strategy will
            be applied to its corresponding output. Use `strategy` for a single
            strategy.
        strategy_weights: List of weights for each strategy. Must have the same
            length as `strategies`. If None, equal weights are used.
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
    student = hub.models.CausalLM.from_preset(
        "gemma2_2b_en", load_weights=False
    )

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

    # Access the trained student model
    trained_student = distiller.student_model
    ```

    For multi-output models:

    ```python
    # Create multiple strategies for different outputs
    strategies = [
        LogitsDistillation(temperature=3.0, output_index=0),
        LogitsDistillation(temperature=2.0, output_index=1)
    ]
    strategy_weights = [1.0, 0.5]  # Weight classification more heavily

    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategies=strategies,
        strategy_weights=strategy_weights,
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
        strategy=None,
        strategies=None,
        strategy_weights=None,
        student_loss_weight=0.5,
        optimizer="adam",
        student_loss="sparse_categorical_crossentropy",
        metrics=None,
        name="distiller",
        **kwargs,
    ):
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

            # Convert string loss to function using tree.map_structure

        def convert_loss_to_function(loss):
            if isinstance(loss, str):
                loss_fn = keras.losses.get(loss)
                if loss_fn is None:
                    raise ValueError(
                        f"Unknown loss function: '{loss}'. "
                        "Please provide a valid loss function name or instance."
                    )
                return loss_fn
            elif loss is None:
                raise ValueError(
                    "Student loss function cannot be None. "
                    "Please provide a valid 'student_loss' parameter."
                )
            else:
                return loss

        self._student_loss = tree.map_structure(
            convert_loss_to_function, student_loss
        )

        # Handle strategy configuration
        if strategy is not None and strategies is not None:
            raise ValueError(
                "Cannot specify both 'strategy' and 'strategies'. "
                "Use 'strategy' for single strategy or 'strategies' for "
                "multiple strategies."
            )

        if strategy is not None:
            # Single strategy mode
            self.strategies = [strategy]
            self.strategy_weights = [1.0]
            self.single_strategy = True
        elif strategies is not None:
            # Multiple strategies mode
            if not isinstance(strategies, (list, tuple)):
                raise ValueError(
                    f"strategies must be a list or tuple, got "
                    f"{type(strategies)}"
                )

            self.strategies = strategies

            # Set default weights if not provided
            if strategy_weights is None:
                self.strategy_weights = [1.0] * len(strategies)
            else:
                if len(strategy_weights) != len(strategies):
                    raise ValueError(
                        f"Number of strategy_weights ({len(strategy_weights)}) "
                        f"must match number of strategies ({len(strategies)})"
                    )
                self.strategy_weights = strategy_weights

            self.single_strategy = False
        else:
            raise ValueError(
                "Must specify either 'strategy' or 'strategies'. "
                "Please provide a valid strategy such as LogitsDistillation, "
                "FeatureDistillation, or a list of strategies."
            )

        # Validate strategy-specific compatibility
        for strategy in self.strategies:
            self._validate_strategy_compatibility(teacher, student, strategy)

        # Create efficient multi-layer feature extractors
        self._create_multi_feature_extractors()

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

    def _validate_strategy_compatibility(self, teacher, student, strategy):
        """Validate that the strategy is compatible with the teacher and student
        models."""
        if hasattr(strategy, "validate_model_compatibility"):
            strategy.validate_model_compatibility(teacher, student)

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

    def _create_multi_feature_extractors(self):
        """Create efficient feature extractors that extract all needed features
        in single forward passes.

        This method analyzes all FeatureDistillation strategies to determine
        which layers need feature extraction, then creates models that extract
        all required features in one pass to avoid redundant computation.
        """
        # Collect all layer names needed for feature extraction
        teacher_layer_names = []
        student_layer_names = []

        for strategy in self.strategies:
            if (
                hasattr(strategy, "teacher_layer_name")
                and strategy.teacher_layer_name
            ):
                if strategy.teacher_layer_name not in teacher_layer_names:
                    teacher_layer_names.append(strategy.teacher_layer_name)
            if (
                hasattr(strategy, "student_layer_name")
                and strategy.student_layer_name
            ):
                if strategy.student_layer_name not in student_layer_names:
                    student_layer_names.append(strategy.student_layer_name)

        # Create multi-output feature extractors if needed
        self._teacher_feature_extractor = None
        self._student_feature_extractor = None
        self._teacher_layer_outputs = {}
        self._student_layer_outputs = {}

        if teacher_layer_names:
            try:
                # For Sequential models, use the last layer's output as final
                if isinstance(self.teacher, keras.Sequential):
                    final_output = self.teacher.layers[-1].output
                    inputs = self.teacher.layers[0].input
                else:
                    # For Functional models
                    if (
                        not hasattr(self.teacher, "inputs")
                        or self.teacher.inputs is None
                    ):
                        raise ValueError("Teacher model has no defined inputs")
                    if (
                        not hasattr(self.teacher, "output")
                        or self.teacher.output is None
                    ):
                        raise ValueError("Teacher model has no defined output")
                    final_output = self.teacher.output
                    inputs = self.teacher.inputs

                teacher_outputs = [final_output]  # Always include final output
                teacher_output_names = ["final_output"]

                for layer_name in teacher_layer_names:
                    layer = self.teacher.get_layer(name=layer_name)
                    teacher_outputs.append(layer.output)
                    teacher_output_names.append(layer_name)

                self._teacher_feature_extractor = keras.Model(
                    inputs=inputs,
                    outputs=teacher_outputs,
                    name=f"{self.teacher.name}_multi_feature_extractor",
                )
                self._teacher_output_names = teacher_output_names
            except (ValueError, AttributeError):
                # Fallback to individual extraction for subclassed models
                self._teacher_feature_extractor = None

        if student_layer_names:
            try:
                # For Sequential models, use the last layer's output as final
                if isinstance(self.student, keras.Sequential):
                    final_output = self.student.layers[-1].output
                    inputs = self.student.layers[0].input
                else:
                    # For Functional models
                    if (
                        not hasattr(self.student, "inputs")
                        or self.student.inputs is None
                    ):
                        raise ValueError("Student model has no defined inputs")
                    if (
                        not hasattr(self.student, "output")
                        or self.student.output is None
                    ):
                        raise ValueError("Student model has no defined output")
                    final_output = self.student.output
                    inputs = self.student.inputs

                student_outputs = [final_output]  # Always include final output
                student_output_names = ["final_output"]

                for layer_name in student_layer_names:
                    layer = self.student.get_layer(name=layer_name)
                    student_outputs.append(layer.output)
                    student_output_names.append(layer_name)

                self._student_feature_extractor = keras.Model(
                    inputs=inputs,
                    outputs=student_outputs,
                    name=f"{self.student.name}_multi_feature_extractor",
                )
                self._student_output_names = student_output_names
            except (ValueError, AttributeError):
                # Fallback to individual extraction for subclassed models
                self._student_feature_extractor = None

    def _extract_all_teacher_features(self, x):
        """Extract all teacher features efficiently in a single forward pass.

        Args:
            x: Input data.

        Returns:
            Dict mapping layer names to their outputs, including 'final_output'.
        """
        if self._teacher_feature_extractor is not None:
            # Use efficient multi-output extractor
            feature_outputs = self._teacher_feature_extractor(x, training=False)
            if not isinstance(feature_outputs, (list, tuple)):
                feature_outputs = [feature_outputs]

            # Map outputs to layer names
            features = {}
            for name, output in zip(
                self._teacher_output_names, feature_outputs
            ):
                features[name] = output
            return features
        else:
            # Fallback: just get final output for LogitsDistillation
            return {"final_output": self.teacher(x, training=False)}

    def _extract_all_student_features(self, x, y_pred):
        """Extract all student features efficiently in a single forward pass.

        Args:
            x: Input data.
            y_pred: Student predictions from forward pass (to avoid
                recomputation).

        Returns:
            Dict mapping layer names to their outputs, including 'final_output'.
        """
        if self._student_feature_extractor is not None:
            # Use efficient multi-output extractor
            feature_outputs = self._student_feature_extractor(x, training=True)
            if not isinstance(feature_outputs, (list, tuple)):
                feature_outputs = [feature_outputs]

            # Map outputs to layer names
            features = {}
            for name, output in zip(
                self._student_output_names, feature_outputs
            ):
                features[name] = output
            return features
        else:
            # Fallback: use y_pred for final output to avoid recomputation
            return {"final_output": y_pred}

    def _get_strategy_features(self, strategy, all_features, is_teacher):
        """Get the specific features needed by a strategy from pre-extracted
        features.

        Args:
            strategy: The FeatureDistillation strategy.
            all_features: Dict of all extracted features.
            is_teacher: Whether these are teacher features.

        Returns:
            The specific features needed by this strategy.
        """
        if is_teacher:
            layer_name = strategy.teacher_layer_name or "final_output"
        else:
            layer_name = strategy.student_layer_name or "final_output"

        if layer_name not in all_features:
            raise ValueError(
                f"Layer '{layer_name}' features not found in extracted "
                f"features. Available features: {list(all_features.keys())}"
            )

        return all_features[layer_name]

    @property
    def student_model(self):
        """The trained student model for independent use.

        This property provides access to the student model that has been trained
        through the distillation process. The student model can be used
        independently for inference, further training, or saving.

        Returns:
            keras.Model: The trained student model.

        Example:
            ```python
            # After training the distiller
            distiller.fit(x_train, y_train, epochs=10)

            # Access the trained student model
            trained_student = distiller.student_model

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
                    student_outputs = y_pred

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
                    return keras.ops.mean(
                        keras.ops.square(teacher_outputs - student_outputs)
                    )
            ```
        """
        # Compute student loss using tree operations for dicts, manual for lists
        student_loss = 0.0
        if self.student_loss_weight > 0.0 and y is not None:
            if isinstance(self._student_loss, dict):
                # Dict case - check keys match at runtime (keys can change)
                loss_keys = set(self._student_loss.keys())
                y_keys = set(y.keys())
                pred_keys = set(y_pred.keys())
                if loss_keys != y_keys or y_keys != pred_keys:
                    raise ValueError(
                        f"Keys must match across loss functions, targets, and "
                        f"predictions. Loss keys: {loss_keys}, "
                        f"Target keys: {y_keys}, Prediction keys: {pred_keys}"
                    )

                # Compute losses manually and sum using tree.flatten
                loss_values = {
                    key: self._student_loss[key](y[key], y_pred[key])
                    for key in self._student_loss.keys()
                }
                flat_losses = tree.flatten(loss_values)
                student_loss = keras.ops.sum(keras.ops.stack(flat_losses))
            elif isinstance(self._student_loss, (list, tuple)):
                # List/tuple case - check lengths match at runtime (can change)
                if len(y) != len(y_pred) or len(self._student_loss) != len(y):
                    raise ValueError(
                        f"Number of targets ({len(y)}), predictions "
                        f"({len(y_pred)}), and loss functions "
                        f"({len(self._student_loss)}) must match."
                    )

                # Compute losses manually and sum using tree.flatten
                loss_values = [
                    loss_fn(y_true, y_pred_i)
                    for loss_fn, y_true, y_pred_i in zip(
                        self._student_loss, y, y_pred
                    )
                ]
                flat_losses = tree.flatten(loss_values)
                student_loss = keras.ops.sum(keras.ops.stack(flat_losses))
            else:
                # Single output case
                student_loss = self._student_loss(y, y_pred)

            # Ensure student_loss is a scalar
            if hasattr(student_loss, "shape") and len(student_loss.shape) > 0:
                student_loss = keras.ops.mean(student_loss)

        # Compute distillation loss
        distillation_loss = 0.0
        if self.student_loss_weight < 1.0:
            # Extract all features efficiently in single forward passes
            teacher_features = self._extract_all_teacher_features(x)
            student_features = self._extract_all_student_features(x, y_pred)

            # Apply strategies using pre-extracted features
            for strategy, weight in zip(self.strategies, self.strategy_weights):
                # Get appropriate outputs/features for this strategy
                if hasattr(strategy, "teacher_layer_name"):
                    # FeatureDistillation - use extracted features
                    strategy_teacher_output = self._get_strategy_features(
                        strategy, teacher_features, is_teacher=True
                    )
                    strategy_student_output = self._get_strategy_features(
                        strategy, student_features, is_teacher=False
                    )
                else:
                    # LogitsDistillation - use final model outputs
                    strategy_teacher_output = teacher_features["final_output"]
                    strategy_student_output = y_pred

                # Compute loss for this strategy
                strategy_loss = strategy.compute_loss(
                    strategy_teacher_output, strategy_student_output
                )

                # Apply weight and add to total
                distillation_loss += weight * strategy_loss

            # Ensure distillation_loss is a scalar
            if (
                hasattr(distillation_loss, "shape")
                and len(distillation_loss.shape) > 0
            ):
                distillation_loss = keras.ops.mean(distillation_loss)

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
                "strategies": [
                    serialization_lib.serialize_keras_object(strategy)
                    for strategy in self.strategies
                ],
                "strategy_weights": self.strategy_weights,
                "student_loss_weight": self.student_loss_weight,
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
        config["strategies"] = [
            serialization_lib.deserialize_keras_object(strategy)
            for strategy in config["strategies"]
        ]
        return cls(**config)
