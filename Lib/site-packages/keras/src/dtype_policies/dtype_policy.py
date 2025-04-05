from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state

QUANTIZATION_MODES = ("int8", "float8")


@keras_export(
    [
        "keras.DTypePolicy",
        "keras.dtype_policies.DTypePolicy",
        "keras.mixed_precision.DTypePolicy",  # Legacy
        "keras.mixed_precision.Policy",  # Legacy
    ]
)
class DTypePolicy:
    """A dtype policy for a Keras layer.

    A dtype policy determines a layer's computation and variable dtypes. Each
    layer has a policy. Policies can be passed to the `dtype` argument of layer
    constructors, or a global policy can be set with
    `keras.config.set_dtype_policy`.

    Args:
        name: The policy name, which determines the compute and variable dtypes.
            Can be any dtype name, such as `"float32"` or `"float64"`,
            which causes both the compute and variable dtypes
            will be that dtype.
            Can also be the string `"mixed_float16"` or `"mixed_bfloat16"`,
            which causes the compute dtype to be `float16` or `bfloat16`
            and the variable dtype to be `float32`.

    Typically you only need to interact with dtype policies when using mixed
    precision, which is the use of float16 or bfloat16 for computations and
    float32 for variables. This is why the term `mixed_precision` appears in the
    API name. Mixed precision can be enabled by passing `"mixed_float16"` or
    `"mixed_bfloat16"` to `keras.mixed_precision.set_dtype_policy()`.

    >>> keras.config.set_dtype_policy("mixed_float16")
    >>> layer1 = keras.layers.Dense(10)
    >>> layer1.dtype_policy  # layer1 will automatically use mixed precision
    <DTypePolicy "mixed_float16">
    >>> # Can optionally override layer to use float32
    >>> # instead of mixed precision.
    >>> layer2 = keras.layers.Dense(10, dtype="float32")
    >>> layer2.dtype_policy
    <DTypePolicy "float32">
    >>> # Set policy back to initial float32.
    >>> keras.config.set_dtype_policy('float32')

    In the example above, passing `dtype="float32"` to the layer is
    equivalent to passing
    `dtype=keras.config.DTypePolicy("float32")`.
    In general, passing a dtype policy name to a layer is equivalent
    to passing the corresponding policy, so it is never necessary
    to explicitly construct a `DTypePolicy` object.
    """

    def __init__(self, name=None):
        # Use the global dtype policy if `name` is not specified
        if name is None:
            name = dtype_policy().name
        self._name = name
        self._compute_dtype, self._variable_dtype = self._parse_name(name)
        self._quantization_mode = None

    def _parse_name(self, name):
        """Parses a `DTypePolicy` name into a compute and variable dtype.

        Args:
            name: The name of the policy.

        Returns:
            The `(compute_dtype, variable_dtype)` pair.
        """
        if not isinstance(name, str):
            raise TypeError(
                "'name' must be a string, such as 'mixed_float16'. "
                f"Received: name={name} (of type {type(name)})"
            )
        if name == "mixed_float16":
            return "float16", "float32"
        elif name == "mixed_bfloat16":
            return "bfloat16", "float32"
        try:
            dtype = backend.standardize_dtype(name)
            return dtype, dtype
        except ValueError:
            raise ValueError(
                f"Cannot convert '{name}' to a mixed precision "
                "DTypePolicy. Valid policies include 'mixed_float16', "
                "'mixed_bfloat16', and the name of any float dtype such as "
                "'float32'."
            )

    @property
    def variable_dtype(self):
        """The variable dtype of this policy.

        This is the dtype layers will create their variables in, unless a layer
        explicitly chooses a different dtype. If this is different than
        `DTypePolicy.compute_dtype`, Layers will cast variables to
        the compute dtype to avoid type errors.

        Variable regularizers are run in the variable dtype, not the compute
        dtype.

        Returns:
            The variable dtype of this policy, as a string.
        """
        return self._variable_dtype

    @property
    def compute_dtype(self):
        """The compute dtype of this policy.

        This is the dtype layers will do their computations in. Typically layers
        output tensors with the compute dtype as well.

        Note that even if the compute dtype is float16 or bfloat16, hardware
        devices may not do individual adds, multiplies, and other fundamental
        operations in float16 or bfloat16, but instead may do some of them in
        float32 for numeric stability. The compute dtype is the dtype of the
        inputs and outputs of the ops that the layer executes.
        Internally, many ops will do certain internal calculations in
        float32 or some other device-internal intermediate format with higher
        precision than float16/bfloat16, to increase numeric stability.

        Returns:
            The compute dtype of this policy, as a string.
        """
        return self._compute_dtype

    @property
    def name(self):
        """Returns the name of this policy."""
        return self._name

    @property
    def quantization_mode(self):
        """The quantization mode of this policy.

        Returns:
            The quantization mode of this policy, as a string. If this policy is
            not quantized, it will return `None`.
        """
        return self._quantization_mode

    def convert_input(self, x, autocast, dtype):
        """Converts the input dtype based on `autocast` and `dtype`.

        Note that `x` can be a tensor, symbolic tensor or numpy array, and this
        method will keep integer inputs untouched and only apply casting to
        floats.
        """

        dtype = backend.standardize_dtype(dtype)
        if backend.is_tensor(x):
            if self._should_cast(x, autocast, dtype):
                x = backend.cast(x, dtype=dtype)
            return x
        elif backend.is_keras_tensor(x):
            if self._should_cast(x, autocast, dtype):
                x = ops.cast(x, dtype=dtype)
            return x
        elif hasattr(x, "__array__"):
            try:
                x = backend.convert_to_tensor(x)
            except TypeError:
                x = backend.convert_to_tensor(x, dtype=dtype)
            if self._should_cast(x, autocast, dtype):
                x = backend.cast(x, dtype=dtype)
            return x
        return x

    def get_config(self):
        return {"name": self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __repr__(self):
        class_name = self.__class__.__name__
        if class_name == "FloatDTypePolicy":
            class_name = "DTypePolicy"
        return f'<{class_name} "{self._name}">'

    def __eq__(self, other):
        if self.__class__ in (DTypePolicy, FloatDTypePolicy):
            if type(other) not in (DTypePolicy, FloatDTypePolicy):
                return False
        else:
            if type(other) is not self.__class__:
                return False
        return self._name == other._name

    def _should_cast(self, x, autocast, dtype):
        x_dtype = backend.standardize_dtype(x.dtype)
        if autocast and backend.is_float_dtype(x_dtype) and x_dtype != dtype:
            return True
        else:
            return False


@keras_export(
    ["keras.FloatDTypePolicy", "keras.dtype_policies.FloatDTypePolicy"]
)
class FloatDTypePolicy(DTypePolicy):
    # An alias for `DTypePolicy`
    pass


@keras_export("keras.dtype_policies.QuantizedDTypePolicy")
class QuantizedDTypePolicy(DTypePolicy):
    def __init__(self, mode, source_name=None):
        # Use the global dtype policy if `source_name` is not specified
        if source_name is None:
            source_name = dtype_policy().name
        name = f"{mode}_from_{source_name}"
        self._compute_dtype, self._variable_dtype = self._parse_name(
            source_name
        )
        self._check_quantization_mode(mode, self._compute_dtype)

        self._name = name
        self._source_name = source_name
        self._quantization_mode = mode

    def __eq__(self, other):
        if super().__eq__(other) is False:
            return False
        return (
            self._quantization_mode == other._quantization_mode
            and self._source_name == other._source_name
        )

    def get_config(self):
        return {
            "mode": self._quantization_mode,
            "source_name": self._source_name,
        }

    def _check_quantization_mode(self, mode, compute_dtype):
        if mode not in QUANTIZATION_MODES:
            raise ValueError(
                "Invalid quantization mode. "
                f"Expected one of {QUANTIZATION_MODES}. "
                f"Received: mode={mode}"
            )
        if compute_dtype == "float16" and mode == "int8":
            raise ValueError(
                f"Quantization mode='{mode}' doesn't work well with "
                "compute_dtype='float16'."
            )


@keras_export("keras.dtype_policies.QuantizedFloat8DTypePolicy")
class QuantizedFloat8DTypePolicy(QuantizedDTypePolicy):
    default_amax_history_length = 1024

    def __init__(self, mode, source_name=None, amax_history_length=1024):
        super().__init__(mode=mode, source_name=source_name)
        if not isinstance(amax_history_length, int):
            raise TypeError(
                "`amax_history_length` must be an integer. "
                f"Received: amax_history_length={amax_history_length}"
            )
        self._amax_history_length = amax_history_length

    @property
    def amax_history_length(self):
        """The length of the amax history window.

        This property is used for scaling factor computation in float8 training.
        """
        return self._amax_history_length

    def __eq__(self, other):
        if super().__eq__(other) is False:
            return False
        return self._amax_history_length == other._amax_history_length

    def get_config(self):
        config = super().get_config()
        config.update({"amax_history_length": self.amax_history_length})
        return config


@keras_export(
    [
        "keras.config.set_dtype_policy",
        "keras.mixed_precision.set_dtype_policy",  # Legacy
        "keras.mixed_precision.set_global_policy",  # Legacy
    ]
)
def set_dtype_policy(policy):
    """Sets the default dtype policy globally.

    Example:

    >>> keras.config.set_dtype_policy("mixed_float16")
    """
    if not isinstance(policy, DTypePolicy):
        if isinstance(policy, str):
            if policy.startswith(QUANTIZATION_MODES):
                policy = _get_quantized_dtype_policy_by_str(policy)
            else:
                policy = DTypePolicy(policy)
        else:
            raise ValueError(
                "Invalid `policy` argument. "
                "Expected the string name of a policy "
                "(such as 'mixed_float16') or a `DTypePolicy` "
                f"instance. Received: policy={policy} "
                f"(of type {type(policy)})"
            )
    global_state.set_global_attribute("dtype_policy", policy)


@keras_export(
    [
        "keras.config.dtype_policy",
        "keras.mixed_precision.dtype_policy",  # Legacy
        "keras.mixed_precision.global_policy",  # Legacy
    ]
)
def dtype_policy():
    """Returns the current default dtype policy object."""
    policy = global_state.get_global_attribute("dtype_policy", None)
    if policy is None:
        policy = DTypePolicy(backend.floatx())
        set_dtype_policy(policy)
    return policy


def _get_quantized_dtype_policy_by_str(policy):
    if not isinstance(policy, str):
        raise TypeError(f"`policy` must be a string. Received: policy={policy}")
    if not policy.startswith(QUANTIZATION_MODES):
        raise ValueError(
            "`policy` is incompatible with the current supported quantization."
        )
    split_name = policy.split("_from_")
    if len(split_name) != 2:
        raise ValueError(
            "Cannot convert `policy` into a valid pair (`mode`, `source_name`) "
            "to instantiate `QuantizedDTypePolicy`. "
            f"Received: policy={policy}"
        )
    mode, source_name = split_name
    if policy.startswith("int8"):
        return QuantizedDTypePolicy(mode, source_name)
    elif policy.startswith("float8"):
        return QuantizedFloat8DTypePolicy(mode, source_name)
    else:
        raise NotImplementedError
