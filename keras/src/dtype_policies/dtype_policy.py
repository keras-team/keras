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

    def __new__(cls, name, *args, **kwargs):
        if not isinstance(name, str):
            raise TypeError(
                "'name' must be a string, such as 'mixed_float16'. "
                f"Received: name={name} (of type {type(name)})"
            )
        # For backwards compatibility
        # TODO: We should consider deprecating this behavior
        if cls is __class__:
            if name.startswith(QUANTIZATION_MODES):
                return _get_quantized_dtype_policy_by_str(name)
            return FloatDTypePolicy(name)
        return super().__new__(cls)

    def __getnewargs__(self):
        # To support `copy`, `deepcopy` and `pickle`
        return (self._name,)

    def __init__(self, name):
        self._name = name
        self._compute_dtype = backend.floatx()
        self._variable_dtype = backend.floatx()

    def _parse_name(self, name):
        """Parses a `DTypePolicy` name into a compute and variable dtype.

        Args:
            name: The name of the policy.

        Returns:
            The `(compute_dtype, variable_dtype)` pair.
        """
        raise NotImplementedError

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

    def convert_input(self, x, autocast, dtype):
        dtype = backend.standardize_dtype(dtype)
        if backend.is_tensor(x):
            if (
                autocast
                and backend.is_float_dtype(x.dtype)
                and x.dtype != dtype
            ):
                x = backend.cast(x, dtype=dtype)
            return x
        elif backend.is_keras_tensor(x):
            if (
                autocast
                and backend.is_float_dtype(x.dtype)
                and x.dtype != dtype
            ):
                x.dtype = dtype
            return x
        elif hasattr(x, "__array__"):
            return ops.convert_to_tensor(x, dtype=dtype)
        return x

    def get_config(self):
        return {"name": self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras_export(
    ["keras.FloatDTypePolicy", "keras.dtype_policies.FloatDTypePolicy"]
)
class FloatDTypePolicy(DTypePolicy):
    def __init__(self, name):
        super().__init__(name)
        self._compute_dtype, self._variable_dtype = self._parse_name(name)
        # TODO: check that the current hardware supports the provided
        # dtype policy and raise/warn otherwise.

    def _parse_name(self, name):
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
                "FloatDTypePolicy. Valid policies include 'mixed_float16', "
                "'mixed_bfloat16', and the name of any float dtype such as "
                "'float32'."
            )

    def __repr__(self):
        return f'<FloatDTypePolicy "{self._name}">'


@keras_export("keras.dtype_policies.QuantizedDTypePolicy")
class QuantizedDTypePolicy(DTypePolicy):
    def __init__(self, name):
        super().__init__(name)
        self._quantization_mode, self._compute_dtype, self._variable_dtype = (
            self._parse_name(name)
        )

    def _parse_name(self, name):
        error_msg = (
            f"Cannot convert '{name}' to a {self.__class__.__name__}. "
            f"Valid policies are: {self._get_all_valid_policies()}."
        )
        split_name = name.split("_from_")
        if len(split_name) != 2:
            raise ValueError(error_msg)
        mode, from_name = split_name
        if mode not in QUANTIZATION_MODES:
            raise ValueError(error_msg)
        if from_name == "mixed_float16" and mode != "int8":
            return mode, "float16", "float32"
        elif from_name == "mixed_bfloat16":
            return mode, "bfloat16", "float32"
        try:
            dtype = backend.standardize_dtype(from_name)
            if dtype == "float16" and mode == "int8":
                raise ValueError
            return mode, dtype, dtype
        except ValueError:
            raise ValueError(error_msg)

    @property
    def quantization_mode(self):
        """The quantization mode of this policy.

        Returns:
            The quantization mode of this policy, as a string.
        """
        return self._quantization_mode

    def __repr__(self):
        return f'<QuantizedDTypePolicy "{self._name}">'

    def _get_all_valid_policies(self):
        valid_float_policies = [
            "float32",
            "float16",
            "bfloat16",
            "mixed_float16",
            "mixed_bfloat16",
        ]
        valid_policies = [
            f"{mode}_from_{policy}"
            for mode in ("int8",)
            for policy in valid_float_policies
        ]
        # Remove invalid policies
        valid_policies.remove("int8_from_float16")
        valid_policies.remove("int8_from_mixed_float16")
        return valid_policies


@keras_export("keras.dtype_policies.QuantizedFloat8DTypePolicy")
class QuantizedFloat8DTypePolicy(QuantizedDTypePolicy):
    def __init__(self, name, amax_history_length=1024):
        super().__init__(name)
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

    def __repr__(self):
        return f'<QuantizedFloat8DTypePolicy "{self._name}">'

    def _get_all_valid_policies(self):
        valid_float_policies = [
            "float32",
            "float16",
            "bfloat16",
            "mixed_float16",
            "mixed_bfloat16",
        ]
        valid_policies = [
            f"{mode}_from_{policy}"
            for mode in ("float8")
            for policy in valid_float_policies
        ]
        return valid_policies

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
                policy = FloatDTypePolicy(policy)
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
        policy = FloatDTypePolicy(backend.floatx())
        set_dtype_policy(policy)
    return policy


def _get_quantized_dtype_policy_by_str(policy):
    if not isinstance(policy, str):
        raise TypeError(f"`policy` must be a string. Received: policy={policy}")
    if not policy.startswith(QUANTIZATION_MODES):
        raise ValueError(
            "`policy` is incompatible with the current supported quantization."
        )
    if policy.startswith("int8"):
        return QuantizedDTypePolicy(policy)
    elif policy.startswith("float8"):
        return QuantizedFloat8DTypePolicy(policy)
    else:
        raise NotImplementedError
