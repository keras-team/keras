import copy
import json

try:
    import difflib
except ImportError:
    difflib = None

from keras.src.api_export import keras_export


@keras_export("keras.utils.Config")
class Config:
    """A Config is a dict-like container for named values.

    It offers a few advantages over a plain dict:

    - Setting and retrieving values via attribute setting / getting.
    - Ability to freeze the config to ensure no accidental config modifications
        occur past a certain point in your program.
    - Easy serialization of the whole config as JSON.

    Examples:

    ```python
    # Create a config via constructor arguments
    config = Config("learning_rate"=0.1, "momentum"=0.9)

    # Then keep adding to it via attribute-style setting
    config.use_ema = True
    config.ema_overwrite_frequency = 100

    # You can also add attributes via dict-like access
    config["seed"] = 123

    # You can retrieve entries both via attribute-style
    # access and dict-style access
    assert config.seed == 100
    assert config["learning_rate"] == 0.1
    ```

    A config behaves like a dict:

    ```python
    config = Config("learning_rate"=0.1, "momentum"=0.9)
    for k, v in config.items():
        print(f"{k}={v}")

    print(f"keys: {list(config.keys())}")
    print(f"values: {list(config.values())}")
    ```

    In fact, it can be turned into one:

    ```python
    config = Config("learning_rate"=0.1, "momentum"=0.9)
    dict_config = config.as_dict()
    ```

    You can easily serialize a config to JSON:

    ```python
    config = Config("learning_rate"=0.1, "momentum"=0.9)

    json_str = config.to_json()
    ```

    You can also freeze a config to prevent further changes:

    ```python
    config = Config()
    config.optimizer = "adam"
    config.seed = 123

    # Freeze the config to prevent changes.
    config.freeze()
    assert config.frozen

    config.foo = "bar"  # This will raise an error.
    ```
    """

    __attrs__ = None

    def __init__(self, **kwargs):
        self._config = kwargs
        self._frozen = False
        self.__attrs__ = set(dir(self))

    @property
    def frozen(self):
        """Returns True if the config is frozen."""
        return self._frozen

    def freeze(self):
        """Marks the config as frozen, preventing any ulterior modification."""
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def _raise_if_frozen(self):
        if self._frozen:
            raise ValueError(
                "Cannot mutate attribute(s) because the config is frozen."
            )

    def as_dict(self):
        return copy.copy(self._config)

    def to_json(self):
        return json.dumps(self._config)

    def keys(self):
        return self._config.keys()

    def values(self):
        return self._config.values()

    def items(self):
        return self._config.items()

    def pop(self, *args):
        self._raise_if_frozen()
        return self._config.pop(*args)

    def update(self, *args, **kwargs):
        self._raise_if_frozen()
        return self._config.update(*args, **kwargs)

    def get(self, keyname, value=None):
        return self._config.get(keyname, value)

    def __setattr__(self, name, value):
        attrs = object.__getattribute__(self, "__attrs__")
        if attrs is None or name in attrs:
            return object.__setattr__(self, name, value)

        self._raise_if_frozen()
        self._config[name] = value

    def __getattr__(self, name):
        attrs = object.__getattribute__(self, "__attrs__")
        if attrs is None or name in attrs:
            return object.__getattribute__(self, name)

        if name in self._config:
            return self._config[name]

        msg = f"Unknown attribute: '{name}'."
        if difflib is not None:
            closest_matches = difflib.get_close_matches(
                name, self._config.keys(), n=1, cutoff=0.7
            )
            if closest_matches:
                msg += f" Did you mean '{closest_matches[0]}'?"
        raise AttributeError(msg)

    def __setitem__(self, key, item):
        self._raise_if_frozen()
        self._config[key] = item

    def __getitem__(self, key):
        return self._config[key]

    def __repr__(self):
        return f"<Config {self._config}>"

    def __iter__(self):
        keys = sorted(self._config.keys())
        for k in keys:
            yield k

    def __len__(self):
        return len(self._config)

    def __delitem__(self, key):
        self._raise_if_frozen()
        del self._config[key]

    def __contains__(self, item):
        return item in self._config
