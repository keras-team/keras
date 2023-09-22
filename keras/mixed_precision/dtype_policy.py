from keras import backend
from keras.api_export import keras_export
from keras.backend.common import global_state


@keras_export(
    [
        "keras.mixed_precision.DTypePolicy",
        "keras.mixed_precision.Policy",
    ]
)
class DTypePolicy:
    """A dtype policy for a Keras layer.

    A dtype policy determines a layer's computation and variable dtypes. Each
    layer has a policy. Policies can be passed to the `dtype` argument of layer
    constructors, or a global policy can be set with
    `keras.mixed_precision.set_dtype_policy`.

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

    >>> keras.mixed_precision.set_dtype_policy("mixed_float16")
    >>> layer1 = keras.layers.Dense(10)
    >>> layer1.dtype_policy  # layer1 will automatically use mixed precision
    <DTypePolicy "mixed_float16">
    >>> # Can optionally override layer to use float32
    >>> # instead of mixed precision.
    >>> layer2 = keras.layers.Dense(10, dtype="float32")
    >>> layer2.dtype_policy
    <DTypePolicy "float32">
    >>> # Set policy back to initial float32.
    >>> keras.mixed_precision.set_dtype_policy('float32')

    In the example above, passing `dtype="float32"` to the layer is
    equivalent to passing
    `dtype=keras.mixed_precision.DTypePolicy("float32")`.
    In general, passing a dtype policy name to a layer is equivalent
    to passing the corresponding policy, so it is never necessary
    to explicitly construct a `DTypePolicy` object.
    """

    def __init__(self, name):
        if not isinstance(name, str):
            raise TypeError(
                "'name' must be a string, such as 'mixed_float16'. "
                f"Received: name={name} (of type {type(name)})"
            )
        self._name = name
        self._compute_dtype, self._variable_dtype = self._parse_name(name)
        # TODO: check that the current hardware supports the provided
        # dtype policy and raise/warn otherwise.

    def _parse_name(self, name):
        """Parses a `DTypePolicy` name into a compute and variable dtype.

        Args:
            name: The name of the policy.

        Returns:
            The `(compute_dtype, variable_dtype)` pair.
        """
        if name == "mixed_float16":
            return "float16", "float32"
        elif name == "mixed_bfloat16":
            return "bfloat16", "float32"
        try:
            dtype = backend.standardize_dtype(name)
            return dtype, dtype
        except ValueError:
            raise ValueError(
                f"Cannot convert '{name}' to a mixed precision DTypePolicy."
                " Valid policies include 'mixed_float16', 'mixed_bfloat16', "
                "and the name of any dtype such as 'float32'."
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

    def __repr__(self):
        return f'<DTypePolicy "{self._name}">'

    def get_config(self):
        return {"name": self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras_export(
    [
        "keras.mixed_precision.set_dtype_policy",
        "keras.mixed_precision.set_global_policy",
    ]
)
def set_dtype_policy(policy):
    """Sets the default dtype policy globally.

    Example:

    >>> keras.mixed_precision.set_dtype_policy("mixed_float16")
    """
    if not isinstance(policy, DTypePolicy):
        if isinstance(policy, str):
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
        "keras.mixed_precision.dtype_policy",
        "keras.mixed_precision.global_policy",
    ]
)
def dtype_policy():
    """Returns the current default dtype policy object."""
    policy = global_state.get_global_attribute("dtype_policy", None)
    if policy is None:
        policy = DTypePolicy(backend.floatx())
        set_dtype_policy(policy)
    return policy
