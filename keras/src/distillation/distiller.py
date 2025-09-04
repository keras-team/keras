import keras
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.models.model import Model
from keras.src.saving import serialization_lib


@keras_export("keras.distillation.Distiller")
class Distiller(Model):
    """Distillation model for transferring knowledge from teacher to student.

    Knowledge distillation transfers knowledge from a large, complex model
    (teacher) to a smaller, simpler model (student). The student learns
    from both ground truth labels and the teacher's predictions, often
    achieving better performance than training on labels alone.

    Args:
        teacher: A trained `keras.Model` that serves as the knowledge source.
            The teacher model is frozen during distillation.
        student: A `keras.Model` to be trained through distillation.
        strategies: List of distillation strategies to apply. Can be a single
            strategy or a list of strategies like `LogitsDistillation`,
            `FeatureDistillation`, or custom distillation strategies.
        strategy_weights: List of weights for each distillation strategy. Must
            have the same length as `strategies`. If None, equal weights used.
        student_loss_weight: Weight for the student's supervised loss component.
            Must be between 0 and 1. Defaults to 0.5.
        optimizer: Optimizer for training the student model. Can be a string
            identifier (e.g., `'adam'`) or an optimizer instance.
        student_loss: Loss function for the student's supervised learning
            component. Can be a string identifier or a loss function instance.
        metrics: List of metrics to track during training.
        name: Name for the distiller model. Defaults to `"distiller"`.
        **kwargs: Additional keyword arguments passed to the parent `Model`
            class.

    Examples:

    ```python
    # Basic distillation with KerasHub models
    import keras_hub as hub

    teacher = hub.models.CausalLM.from_preset("gemma_2b_en")
    student = hub.models.CausalLM.from_preset(
        "gemma_1.1_2b_en", load_weights=False
    )

    # Single distillation strategy
    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategies=LogitsDistillation(temperature=3.0),
        optimizer='adam',
        student_loss='sparse_categorical_crossentropy'
    )

    # Train the distiller
    distiller.fit(x_train, y_train, epochs=10)

    # Access the trained student model
    trained_student = distiller.student_model

    # Multiple distillation strategies
    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategies=[
            LogitsDistillation(temperature=3.0),
            FeatureDistillation(
                teacher_layer_name="dense_1",
                student_layer_name="dense_1"
            )
        ],
        strategy_weights=[1.0, 0.5],
        optimizer='adam',
        student_loss='sparse_categorical_crossentropy'
    )
    ```
    """

    def __init__(
        self,
        teacher,
        student,
        strategies,
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

        # Handle strategies configuration
        if strategies is None:
            raise ValueError(
                "Must specify 'strategies'. "
                "Please provide a valid distillation strategy such as "
                "LogitsDistillation, FeatureDistillation, or a list."
            )

        # Convert single strategy to list for uniform handling
        if not isinstance(strategies, (list, tuple)):
            self.strategies = [strategies]
            self.strategy_weights = [1.0]
        else:
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

        # Validate strategy-specific compatibility and create feature extractors
        for strategy in self.strategies:
            self._validate_strategy_compatibility(teacher, student, strategy)

        self._create_multi_feature_extractors()

        # Initialize the model - compile with provided parameters
        self.compile(optimizer=optimizer, loss=student_loss, metrics=metrics)

        # Freeze teacher model
        self.teacher.trainable = False

        # Initialize loss tracking metrics
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = keras.metrics.Mean(
            name="distillation_loss"
        )
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    def _validate_models(self, teacher, student):
        """Validate that teacher and student models are compatible."""
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
        """Create feature extractors for efficient multi-layer extraction."""
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

        if teacher_layer_names:
            self._create_feature_extractor(
                self.teacher, teacher_layer_names, "teacher"
            )

        if student_layer_names:
            self._create_feature_extractor(
                self.student, student_layer_names, "student"
            )

    def _create_feature_extractor(self, model, layer_names, model_type):
        """Create feature extractor for a model."""
        try:
            # Get model inputs and final output
            if isinstance(model, keras.Sequential):
                final_output = model.layers[-1].output
                inputs = model.layers[0].input
            else:
                if not hasattr(model, "inputs") or model.inputs is None:
                    raise ValueError(
                        f"{model_type} model has no defined inputs"
                    )
                if not hasattr(model, "output") or model.output is None:
                    raise ValueError(
                        f"{model_type} model has no defined output"
                    )
                final_output = model.output
                inputs = model.inputs

            # Collect outputs
            outputs = [final_output]  # Always include final output
            output_names = ["final_output"]

            for layer_name in layer_names:
                layer = model.get_layer(name=layer_name)
                outputs.append(layer.output)
                output_names.append(layer_name)

            # Create extractor
            extractor = keras.Model(
                inputs=inputs,
                outputs=outputs,
                name=f"{model.name}_multi_feature_extractor",
            )

            # Store based on model type
            if model_type == "teacher":
                self._teacher_feature_extractor = extractor
                self._teacher_output_names = output_names
            else:
                self._student_feature_extractor = extractor
                self._student_output_names = output_names

        except (ValueError, AttributeError):
            # Fallback for subclassed models
            if model_type == "teacher":
                self._teacher_feature_extractor = None
            else:
                self._student_feature_extractor = None

    def _extract_all_teacher_features(self, x):
        """Extract all teacher features in a single forward pass.

        Args:
            x: Input data.

        Returns:
            Dict mapping layer names to their outputs.
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
        """Extract all student features in a single forward pass.

        Args:
            x: Input data.
            y_pred: Student predictions from forward pass.

        Returns:
            Dict mapping layer names to their outputs.
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
        """Get the specific features needed by a strategy.

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

    def compile(self, optimizer="adam", loss=None, metrics=None, **kwargs):
        """Compile the distiller with proper integration.

        Args:
            optimizer: Optimizer for training the student model.
            loss: Student loss function (stored for serialization).
            metrics: Additional metrics to track during training.
            **kwargs: Additional arguments passed to parent compile.
        """
        # Store the student loss for serialization (not used in training)
        self._student_loss_for_serialization = loss

        # Compile with a dummy loss since we override compute_loss
        super().compile(
            optimizer=optimizer,
            loss=None,  # We handle loss in compute_loss
            metrics=metrics,
            **kwargs,
        )

    @property
    def student_model(self):
        """The trained student model for independent use.

        Returns:
            keras.Model: The trained student model.
        """
        return self.student

    def call(self, inputs, training=None, **kwargs):
        """Forward pass returns student predictions."""
        return self.student(inputs, training=training, **kwargs)

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, training=True
    ):
        """Compute combined distillation loss.

        Args:
            x: Input data.
            y: Target data.
            y_pred: Model predictions.
            sample_weight: Sample weights (currently unused).
            training: Whether the model is in training mode.

        Returns:
            Combined loss tensor.
        """
        # Handle case where y_pred is not provided
        if y_pred is None:
            y_pred = self(x, training=training)
        # Compute student loss using tree operations for dicts, manual for lists
        student_loss = 0.0
        if self.student_loss_weight > 0.0 and y is not None:
            if isinstance(self._student_loss, dict):
                # Dict case - check keys match at runtime
                loss_keys = set(self._student_loss.keys())
                y_keys = set(y.keys())
                pred_keys = set(y_pred.keys())
                if loss_keys != y_keys or y_keys != pred_keys:
                    raise ValueError(
                        f"Keys must match across loss functions, targets, and "
                        f"predictions. Loss keys: {loss_keys}, "
                        f"Target keys: {y_keys}, Prediction keys: {pred_keys}"
                    )

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
            teacher_features = self._extract_all_teacher_features(x)
            student_features = self._extract_all_student_features(x, y_pred)

            # Apply strategies using pre-extracted features
            for strategy, weight in zip(self.strategies, self.strategy_weights):
                # Get appropriate outputs/features for this strategy
                if (
                    hasattr(strategy, "teacher_layer_name")
                    and strategy.teacher_layer_name is not None
                ):
                    # FeatureDistillation with specific layers
                    try:
                        strategy_teacher_output = self._get_strategy_features(
                            strategy, teacher_features, is_teacher=True
                        )
                        strategy_student_output = self._get_strategy_features(
                            strategy, student_features, is_teacher=False
                        )
                    except ValueError as e:
                        # Provide more helpful error message for feature
                        # extraction failures
                        raise RuntimeError(
                            f"FeatureDistillation failed for strategy "
                            f"targeting teacher layer "
                            f"'{strategy.teacher_layer_name}' and student "
                            f"layer '{strategy.student_layer_name}'. This can "
                            f"happen with subclassed models that haven't "
                            f"been built properly. Consider using only "
                            f"LogitsDistillation for such models. "
                            f"Original error: {e}"
                        ) from e
                else:
                    # LogitsDistillation or FeatureDistillation (final outputs)
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
        # Get parent metrics (from compile)
        parent_metrics = []
        if hasattr(super(), "metrics"):
            parent_metrics = [
                m
                for m in super().metrics
                if m
                not in [
                    self.total_loss_tracker,
                    self.student_loss_tracker,
                    self.distillation_loss_tracker,
                ]
            ]

        # Add our custom loss trackers first
        distillation_metrics = [
            self.total_loss_tracker,
            self.student_loss_tracker,
            self.distillation_loss_tracker,
        ]
        return distillation_metrics + parent_metrics

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
                # Save current state, not initial parameters
                "optimizer": serialization_lib.serialize_keras_object(
                    self.optimizer
                )
                if hasattr(self, "optimizer") and self.optimizer
                else None,
                "student_loss": serialization_lib.serialize_keras_object(
                    getattr(self, "_student_loss_for_serialization", None)
                ),
                # Note: metrics are not easily serializable due to
                # CompileMetrics complexity, so we skip them in serialization
                "metrics": None,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        config = config.copy()

        # Deserialize objects
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

        # Handle optional parameters
        if "optimizer" in config and config["optimizer"] is not None:
            config["optimizer"] = serialization_lib.deserialize_keras_object(
                config["optimizer"]
            )
        if "student_loss" in config and config["student_loss"] is not None:
            config["student_loss"] = serialization_lib.deserialize_keras_object(
                config["student_loss"]
            )
        if "metrics" in config and config["metrics"] is not None:
            config["metrics"] = serialization_lib.deserialize_keras_object(
                config["metrics"]
            )

        return cls(**config)
