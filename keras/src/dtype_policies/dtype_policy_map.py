import re
from collections.abc import MutableMapping

from keras.src import dtype_policies
from keras.src.api_export import keras_export


@keras_export(["keras.dtype_policies.DTypePolicyMap"])
class DTypePolicyMap(MutableMapping):
    def __init__(self, default_policy=None, policy_map=None):
        if policy_map is not None and not isinstance(policy_map, dict):
            raise TypeError(
                "If specified, `policy_map` must be a dict. "
                f"Received: policy_map={policy_map} of type {type(policy_map)}"
            )
        # If `default_policy` is not specified, the policy from
        # `keras.config.dtype_policy` will be used.
        if default_policy is not None:
            default_policy = dtype_policies.get(default_policy)
        self._default_policy = default_policy
        self._policy_map = policy_map or dict()

    @property
    def default_policy(self):
        return self._default_policy

    def __getitem__(self, key):
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
        return self._default_policy

    def __setitem__(self, key, policy):
        if key in self._policy_map:
            raise ValueError(
                f"{key} already exist in the DTypePolicyMap with "
                f"value {self._policy_map[key]}. Please make sure to "
                "not use duplicated keys."
            )
        policy = dtype_policies.get(policy)
        self._policy_map[key] = policy

    def __delitem__(self, key):
        # Let the dict to handle the key missing error
        return self._policy_map.pop(key)

    def get_config(self):
        policy_map = self._policy_map
        if self._default_policy is None:
            # `None` means we want to make these policies follow
            # the setting of `keras.config.dtype_policy` during loading.
            # To support this feature, we can set `name` and `source_name` to
            # `None` in `FloatDTypePolicy` and `QuantizedDTypePolicy`,
            # respectively.
            new_policy_map = dict()
            for k, policy in policy_map.items():
                policy_config = dtype_policies.serialize(policy)
                if "name" in policy_config["config"]:
                    policy_config["config"]["name"] = None
                elif "source_name" in policy_config["config"]:
                    policy_config["config"]["source_name"] = None
                new_policy_map[k] = policy_config
            policy_map = new_policy_map
        return {
            "default_policy": self._default_policy,
            "policy_map": policy_map,
        }

    @classmethod
    def from_config(cls, config):
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
