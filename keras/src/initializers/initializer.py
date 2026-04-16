from keras.src.api_export import keras_export


@keras_export(["keras.Initializer", "keras.initializers.Initializer"])
class Initializer:
    """Initializer base class: all Keras initializers inherit from this class.

    Initializers should implement a `__call__()` method with one of two possible
    signatures. If your initializer does not handle distribution layouts, it
    should use this signature:

    ```python
    def __call__(self, shape, dtype=None):
        # returns a tensor of shape `shape` and dtype `dtype`
        # containing values drawn using a function of your choice.
    ```

    If your initializer handles distribution layouts, it should use this
    signature. `layout` can either be a `keras.distribution.TensorLayout` or a
    backend-specific layout. If the given `layout` is not `None`, the returned
    value must be distributed according to `layout`.

    ```python
    def __call__(self, shape, dtype=None, layout=None):
        # returns a tensor of shape `shape`, dtype `dtype` and distributed
        # across devices according to the layout `layout`.
        # containing values drawn using a function of your choice.
    ```

    Optionally, you can also implement the method `get_config()` and the class
    method `from_config` in order to support serialization, just like with
    any Keras object.

    Here's a simple example: a random normal initializer.

    ```python
    class ExampleRandomNormal(Initializer):
        def __init__(self, mean, stddev):
            self.mean = mean
            self.stddev = stddev

        def __call__(self, shape, dtype=None):
            return keras.random.normal(
                shape, mean=self.mean, stddev=self.stddev, dtype=dtype
            )

        def get_config(self):  # To support serialization
            return {"mean": self.mean, "stddev": self.stddev}
    ```

    Note that we don't have to implement `from_config()` in the example above
    since the constructor arguments of the class the keys in the config returned
    by `get_config()` are the same. In this case, the default `from_config()`
    works fine.
    """

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor.
        """
        raise NotImplementedError(
            "Initializer subclasses must implement the `__call__()` method."
        )

    def get_config(self):
        """Returns the initializer's configuration as a JSON-serializable dict.

        Returns:
            A JSON-serializable Python dict.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """Instantiates an initializer from a configuration dictionary.

        Example:

        ```python
        initializer = RandomUniform(-1, 1)
        config = initializer.get_config()
        initializer = RandomUniform.from_config(config)
        ```

        Args:
            config: A Python dictionary, the output of `get_config()`.

        Returns:
            An `Initializer` instance.
        """
        return cls(**config)

    def clone(self):
        return self.__class__.from_config(self.get_config())
