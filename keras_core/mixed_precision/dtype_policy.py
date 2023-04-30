from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.backend import global_state


@keras_core_export("keras_core.mixed_precision.DTypePolicy")
class DTypePolicy:
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


@keras_core_export("keras_core.mixed_precision.set_dtype_policy")
def set_dtype_policy(policy):
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
    global_state.set_global_setting("dtype_policy", policy)


@keras_core_export("keras_core.mixed_precision.dtype_policy")
def dtype_policy():
    policy = global_state.get_global_setting("dtype_policy", None)
    if policy is None:
        policy = DTypePolicy(backend.floatx())
        set_dtype_policy(policy)
    return policy
