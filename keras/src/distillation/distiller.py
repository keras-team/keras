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
        name: Name for the distiller model. Defaults to `"distiller"`.
        **kwargs: Additional keyword arguments passed to the parent `Model`
            class.

    Attributes:
        student: The student model being trained. Access this to get the trained
            student model for independent use after distillation training.
        teacher: The teacher model providing knowledge. This model is frozen
            during training.

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
    )

    # Compile the distiller (like any Keras model)
    distiller.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the distiller
    distiller.fit(x_train, y_train, epochs=10)

    # Access the trained student model
    trained_student = distiller.student

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
    )

    # Compile with custom settings
    distiller.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
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

    def _assert_shapes_are_compatible(self, shape1, shape2, context):
        """Assert that two shapes are compatible (allowing for batch dimension
        flexibility)."""
        # Check if they have the same number of dimensions
        if len(shape1) != len(shape2):
            raise ValueError(
                f"Teacher and student {context} shapes have different number "
                f"of dimensions. Teacher {context} shape: {shape1}, "
                f"Student {context} shape: {shape2}."
            )

        # Check all dimensions (including batch dimension for distillation)
        for dim1, dim2 in zip(shape1, shape2):
            if dim1 is not None and dim2 is not None and dim1 != dim2:
                raise ValueError(
                    f"Teacher and student {context} shapes are incompatible. "
                    f"Teacher {context} shape: {shape1}, "
                    f"Student {context} shape: {shape2}. "
                    f"All dimensions must match for distillation."
                )

    def _assert_same_dtype(self, teacher_dtype, student_dtype, context):
        """Assert that teacher and student dtypes are the same."""
        if teacher_dtype != student_dtype:
            raise ValueError(
                f"Teacher and student {context} dtypes are incompatible. "
                f"Teacher {context} dtype: {teacher_dtype}, "
                f"Student {context} dtype: {student_dtype}. "
                f"Both models must use the same data type."
            )

    def _validate_input_compatibility(self, teacher, student):
        """Validate that teacher and student have compatible input shapes."""
        # If symbolic tensors are not available (subclassed models), skip.
        if not hasattr(teacher, "inputs") or not hasattr(student, "inputs"):
            return
        teacher_inputs = getattr(teacher, "inputs")
        student_inputs = getattr(student, "inputs")
        if teacher_inputs is None or student_inputs is None:
            return

        # Validate input structures and shapes
        tree.map_structure(
            lambda ti, si: self._assert_shapes_are_compatible(
                ti.shape, si.shape, "input"
            ),
            teacher_inputs,
            student_inputs,
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

        # Validate output structures and shapes
        tree.map_structure(
            lambda to, so: self._assert_shapes_are_compatible(
                to.shape, so.shape, "output"
            ),
            teacher_outputs,
            student_outputs,
        )

    def _validate_dtype_compatibility(self, teacher, student):
        """Validate that teacher and student have compatible data types."""
        # If symbolic tensors are not available (subclassed models), skip.
        if not hasattr(teacher, "inputs") or not hasattr(student, "inputs"):
            return
        if teacher.inputs is None or student.inputs is None:
            return

        # Check input dtypes
        tree.map_structure(
            lambda ti, si: self._assert_same_dtype(ti.dtype, si.dtype, "input"),
            teacher.inputs,
            student.inputs,
        )

        # Check output dtypes
        if not hasattr(teacher, "outputs") or not hasattr(student, "outputs"):
            return
        if teacher.outputs is None or student.outputs is None:
            return

        tree.map_structure(
            lambda to, so: self._assert_same_dtype(
                to.dtype, so.dtype, "output"
            ),
            teacher.outputs,
            student.outputs,
        )

    def _validate_strategy_compatibility(self, teacher, student, strategy):
        """Validate that the strategy is compatible with the teacher and student
        models."""
        strategy.validate_model_compatibility(teacher, student)

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
        self._teacher_feature_extractor = self._create_feature_extractor(
            self.teacher, teacher_layer_names
        )
        self._student_feature_extractor = self._create_feature_extractor(
            self.student, student_layer_names
        )

    def _create_feature_extractor(self, model, layer_names):
        """Create a feature extractor for a model.

        Args:
            model: The model to create an extractor for.
            layer_names: List of layer names to extract features from.

        Returns:
            keras.Model: Feature extractor that returns a dict of features,
            or None if no layer names provided or extractor creation fails.
        """
        # Return None if no layer names provided
        if not layer_names:
            return None

        try:
            # Get model inputs and final output
            if isinstance(model, keras.Sequential):
                final_output = model.layers[-1].output
                inputs = model.layers[0].input
            else:
                if not hasattr(model, "inputs") or model.inputs is None:
                    raise ValueError(
                        f"{model.name} model has no defined inputs"
                    )
                if not hasattr(model, "output") or model.output is None:
                    raise ValueError(
                        f"{model.name} model has no defined output"
                    )
                final_output = model.output
                inputs = model.inputs

            # Collect outputs
            outputs = {"final_output": final_output}
            for layer_name in layer_names:
                layer = model.get_layer(name=layer_name)
                outputs[layer_name] = layer.output

            # Create and return extractor
            return keras.Model(
                inputs=inputs,
                outputs=outputs,
                name=f"{model.name}_multi_feature_extractor",
            )
        except (ValueError, AttributeError):
            # Fallback for subclassed models
            return None

    def _extract_all_teacher_features(self, x):
        """Extract all teacher features in a single forward pass.

        Args:
            x: Input data.

        Returns:
            Dict mapping layer names to their outputs.
        """
        if self._teacher_feature_extractor is not None:
            # Use efficient multi-output extractor (returns dict directly)
            return self._teacher_feature_extractor(x, training=False)
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
            # Use efficient multi-output extractor (returns dict directly)
            return self._student_feature_extractor(x, training=True)
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
            loss: Student loss function for the student's supervised learning.
                Can be a string identifier or a loss function instance.
            metrics: Additional metrics to track during training.
            **kwargs: Additional arguments passed to parent compile.
        """
        # Validate and convert student loss
        if loss is None:
            raise ValueError(
                "Student loss function cannot be None. "
                "Please provide a valid 'loss' parameter."
            )

        # Convert string loss to function using tree.map_structure
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

        self._student_loss = tree.map_structure(convert_loss_to_function, loss)

        # Store the student loss for serialization
        self._student_loss_for_serialization = loss

        # Validate metrics parameter
        if metrics is not None and not isinstance(metrics, (list, tuple)):
            raise ValueError(
                f"metrics must be a list or tuple, got {type(metrics)}"
            )

        # Compile with a dummy loss since we override compute_loss
        super().compile(
            optimizer=optimizer,
            loss=None,  # We handle loss in compute_loss
            metrics=metrics,
            **kwargs,
        )

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
            # Use tree.map_structure for cleaner loss computation
            try:
                loss_values = tree.map_structure(
                    lambda l, o, o_pred: l(o, o_pred),
                    self._student_loss,
                    y,
                    y_pred,
                )
                flat_losses = tree.flatten(loss_values)
                student_loss = (
                    keras.ops.sum(keras.ops.stack(flat_losses))
                    if len(flat_losses) > 1
                    else flat_losses[0]
                )
            except (ValueError, TypeError):
                # Fallback for TrackedDict compatibility issues
                if isinstance(self._student_loss, dict):
                    loss_values = {
                        key: self._student_loss[key](y[key], y_pred[key])
                        for key in self._student_loss.keys()
                    }
                    flat_losses = tree.flatten(loss_values)
                    student_loss = keras.ops.sum(keras.ops.stack(flat_losses))
                elif isinstance(self._student_loss, (list, tuple)):
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

                # Validate outputs are compatible for this strategy
                strategy.validate_outputs(
                    strategy_teacher_output, strategy_student_output
                )

                # Compute loss for this strategy
                strategy_loss = strategy.compute_loss(
                    strategy_teacher_output, strategy_student_output
                )

                # Apply weight and add to total
                distillation_loss = keras.ops.add(
                    distillation_loss, keras.ops.multiply(weight, strategy_loss)
                )

            # Ensure distillation_loss is a scalar
            if len(distillation_loss.shape) > 0:
                distillation_loss = keras.ops.mean(distillation_loss)

        # Combine losses
        total_loss = keras.ops.add(
            keras.ops.multiply(self.student_loss_weight, student_loss),
            keras.ops.multiply(
                keras.ops.subtract(1.0, self.student_loss_weight),
                distillation_loss,
            ),
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

        return cls(**config)
