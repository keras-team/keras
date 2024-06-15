import re
from collections.abc import MutableMapping

from keras.src import dtype_policies
from keras.src.api_export import keras_export


@keras_export(["keras.dtype_policies.DTypePolicyMap"])
class DTypePolicyMap(MutableMapping):
    """A dict-like object that maps string to `DTypePolicy` instances.

    `DTypePolicyMap` can be used in `get_config` in layers and subclasses to
    support a complex configurations of dtype policies.

    For example, we can modify `get_config` in `layers.MultiHeadAttention` as
    follows to support the mixing of dtype policies, such as quantization.

    ```python
    @keras.saving.register_keras_serializable("MyPackage")
    class MyMultiHeadAttention(keras.layers.MultiHeadAttention):
        def get_config(self):
            config = super().get_config()
            dtype_policy_map = dtype_policies.DTypePolicyMap()
            for layer in self._flatten_layers():
                if layer.dtype_policy.is_quantized:
                    dtype_policy_map[layer.name] = layer.dtype_policy
            if len(dtype_policy_map) > 0:
                config.update({"dtype": dtype_policy_map})
            return config
    ```

    Internally, `DTypePolicyMap` uses a string as key and a `DTypePolicy`
    as value. There is a behavior difference between a normal Python dict and
    this class. The string key will be treated as a regex when retrieving the
    value. See the docstring of `get` for more details.

    See below for a usage example. You can define the naming schema
    of the `DTypePolicy`, and then retrieve the corresponding `DTypePolicy`
    instance.

    In the normal case, the key to query is usually the `layer.name`, which
    is the `name` of the layer.

    ```python
    dtype_policy_map = DTypePolicyMap()
    dtype_policy_map["layer/dense_0"] = FloatDTypePolicy("bfloat16")
    dtype_policy_map["layer/dense_1"] = QuantizedDTypePolicy("int8", "bfloat16")

    policy_0 = dtype_policy_map["layer/dense_0"]
    policy_1 = dtype_policy_map["layer/dense_1"]
    policy_2 = dtype_policy_map["layer/dense_2"]  # No hit
    assert policy_0 == FloatDTypePolicy("bfloat16")
    assert policy_1 == QuantizedDTypePolicy("int8", "bfloat16")
    assert policy_2 == keras.config.dtype_policy()
    ```

    Args:
        default_policy: An optional `DTypePolicy` instance specifying the
            default dtype policy. If not specified, the value will default to
            `keras.config.dtype_policy()`.
        policy_map: An optional dict that maps string to `DTypePolicy`
            instances. Defaults to `None`
    """

    def __init__(self, default_policy=None, policy_map=None):
        if policy_map is not None and not isinstance(policy_map, dict):
            raise TypeError(
                "If specified, `policy_map` must be a dict. "
                f"Received: policy_map={policy_map} of type {type(policy_map)}"
            )
        # `default_policy=None` enables us to defer to
        # `keras.config.dtype_policy()` during loading.
        if default_policy is not None:
            default_policy = dtype_policies.get(default_policy)
        self._default_policy = default_policy
        self._policy_map = policy_map or dict()

    @property
    def default_policy(self):
        """The default dtype policy.

        If `default_policy` is not specified in the constructor, this property
        will be `keras.config.dtype_policy()`.
        """
        return dtype_policies.get(self._default_policy)

    def __getitem__(self, key):
        """Retrieves the corresponding `DTypePolicy` by the string key.

        When there isn't an exact match, all the existing keys in the map
        will be treated as a regex and map against the input key again. When
        there are multiple matches for the regex, an `ValueError` will be
        raised. Returns `self.default_policy` if there isn't any match found.

        Args:
            key: String key to query a `DTypePolicy`.

        Returns:
            Corresponding `DTypePolicy` based on the query.
        """
        if key in self._policy_map:
            return self._policy_map[key]

        matching_keys = []
        for k in self._policy_map:
            if re.search(k, key):
                matching_keys.append(k)
        if len(matching_keys) > 1:
            raise ValueError(
                f"Path '{key}' matches multiple dtype policy "
                f"specification keys: {matching_keys}. Please make "
                "sure each path only matches at most "
                "one dtype policy specification key in the DTypePolicyMap."
            )
        elif len(matching_keys) == 1:
            return self._policy_map[matching_keys[0]]
        return self.default_policy

    def __setitem__(self, key, policy):
        """Insert `DTypePolicy` to the `DTypePolicyMap`.

        Args:
            key: String key for the `DTypePolicy`.
            policy: The `DTypePolicy`.
        """
        if key in self._policy_map:
            raise ValueError(
                f"{key} already exist in the DTypePolicyMap with "
                f"value {self._policy_map[key]}. Please make sure to "
                "not use duplicated keys."
            )
        try:
            policy = dtype_policies.get(policy)
        except Exception:
            raise ValueError(
                "Cannot interpret the assigned value by "
                "`keras.dtype_policies.get`. "
                f"Received: {policy} of type {type(policy)}"
            )
        self._policy_map[key] = policy

    def __delitem__(self, key):
        # Let the dict to handle the key missing error
        return self._policy_map.pop(key)

    def __contains__(self, key):
        return key in self._policy_map

    def get_config(self):
        from keras.src.saving import serialization_lib

        policy_map = self._policy_map
        if self._default_policy is None:
            # `self._default_policy=None` enables us to defer to
            # `keras.config.dtype_policy()` during loading.
            # To support this feature, we can set `_name` and `_source_name` to
            # `None` in `FloatDTypePolicy` and `QuantizedDTypePolicy`,
            # respectively.
            for policy in policy_map.values():
                if isinstance(policy, dtype_policies.QuantizedDTypePolicy):
                    policy._name = None
                    policy._source_name = None
                elif isinstance(policy, dtype_policies.DTypePolicy):
                    policy._name = None
        return {
            "default_policy": self._default_policy,
            "policy_map": serialization_lib.serialize_keras_object(policy_map),
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from keras.src.saving import serialization_lib

        config = config.copy()
        config["policy_map"] = serialization_lib.deserialize_keras_object(
            config["policy_map"], custom_objects=custom_objects
        )
        return cls(**config)

    def __len__(self):
        return len(self._policy_map)

    def __iter__(self):
        return iter(self._policy_map)

    def __repr__(self):
        default_policy = (
            self._default_policy.name
            if self._default_policy is not None
            else None
        )
        mapping = []
        for k, v in self._policy_map.items():
            mapping.append((k, v.name))
        return (
            f"<DTypePolicyMap at {hex(id(self))} "
            f"default_policy={default_policy}, "
            f"mapping={mapping}>"
        )