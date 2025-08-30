import keras.src.backend
from keras.src import tree
from keras.src.layers.layer import Layer
from keras.src.random.seed_generator import SeedGenerator
from keras.src.utils import backend_utils
from keras.src.utils import jax_utils
from keras.src.utils import tracking


class DataLayer(Layer):
    """Layer designed for safe use in `tf.data` or `grain` pipeline.

    This layer overrides the `__call__` method to ensure that the correct
    backend is used and that computation is performed on the CPU.

    The `call()` method in subclasses should use `self.backend` ops. If
    randomness is needed, define both `seed` and `generator` in `__init__` and
    retrieve the running seed using `self._get_seed_generator()`. If the layer
    has weights in `__init__` or `build()`, use `convert_weight()` to ensure
    they are in the correct backend.

    **Note:** This layer and its subclasses only support a single input tensor.

    Examples:

    **Custom `DataLayer` subclass:**

    ```python
    from keras.src.layers.preprocessing.data_layer import DataLayer
    from keras.src.random import SeedGenerator


    class BiasedRandomRGBToHSVLayer(DataLayer):
        def __init__(self, seed=None, **kwargs):
            super().__init__(**kwargs)
            self.probability_bias = ops.convert_to_tensor(0.01)
            self.seed = seed
            self.generator = SeedGenerator(seed)

        def call(self, inputs):
            images_shape = self.backend.shape(inputs)
            batch_size = 1 if len(images_shape) == 3 else images_shape[0]
            seed = self._get_seed_generator(self.backend._backend)

            probability = self.backend.random.uniform(
                shape=(batch_size,),
                minval=0.0,
                maxval=1.0,
                seed=seed,
            )
            probability = self.backend.numpy.add(
                probability, self.convert_weight(self.probability_bias)
            )
            hsv_images = self.backend.image.rgb_to_hsv(inputs)
            return self.backend.numpy.where(
                probability[:, None, None, None] > 0.5,
                hsv_images,
                inputs,
            )

        def compute_output_shape(self, input_shape):
            return input_shape
    ```

    **Using as a regular Keras layer:**

    ```python
    import numpy as np

    x = np.random.uniform(size=(1, 16, 16, 3)).astype("float32")
    print(BiasedRandomRGBToHSVLayer()(x).shape)  # (1, 16, 16, 3)
    ```

    **Using in a `tf.data` pipeline:**

    ```python
    import tensorflow as tf

    tf_ds = tf.data.Dataset.from_tensors(x)
    tf_ds = tf_ds.map(BiasedRandomRGBToHSVLayer())
    print([x.shape for x in tf_ds])  # [(1, 16, 16, 3)]
    ```

    **Using in a `grain` pipeline:**

    ```python
    import grain

    grain_ds = grain.MapDataset.source([x])
    grain_ds = grain_ds.map(BiasedRandomRGBToHSVLayer())
    print([x.shape for x in grain_ds])  # [(1, 16, 16, 3)]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend_utils.DynamicBackend()
        self._allow_non_tensor_positional_args = True

    def __call__(self, inputs, **kwargs):
        sample_input = tree.flatten(inputs)[0]
        if (
            not isinstance(sample_input, keras.KerasTensor)
            and backend_utils.in_tf_graph()
            and not jax_utils.is_in_jax_tracing_scope(sample_input)
        ):
            # We're in a TF graph, e.g. a tf.data pipeline.
            self.backend.set_backend("tensorflow")
            inputs = tree.map_structure(
                lambda x: self.backend.convert_to_tensor(
                    x, dtype=self.compute_dtype
                ),
                inputs,
            )
            switch_convert_input_args = False
            if self._convert_input_args:
                self._convert_input_args = False
                switch_convert_input_args = True
            try:
                outputs = super().__call__(inputs, **kwargs)
            finally:
                self.backend.reset()
                if switch_convert_input_args:
                    self._convert_input_args = True
            return outputs
        elif (
            not isinstance(sample_input, keras.KerasTensor)
            and backend_utils.in_grain_data_pipeline()
        ):
            # We're in a Grain data pipeline. Force computation and data
            # placement to CPU.
            with keras.src.backend.device_scope("cpu"):
                return super().__call__(inputs, **kwargs)
        else:
            return super().__call__(inputs, **kwargs)

    @tracking.no_automatic_dependency_tracking
    def _get_seed_generator(self, backend=None):
        if not hasattr(self, "seed") or not hasattr(self, "generator"):
            raise ValueError(
                "The `seed` and `generator` variable must be set in the "
                "`__init__` method before calling `_get_seed_generator()`."
            )
        if backend is None or backend == keras.backend.backend():
            return self.generator
        if not hasattr(self, "_backend_generators"):
            self._backend_generators = {}
        if backend in self._backend_generators:
            return self._backend_generators[backend]
        seed_generator = SeedGenerator(self.seed, backend=self.backend)
        self._backend_generators[backend] = seed_generator
        return seed_generator

    def convert_weight(self, weight):
        """Convert the weight if it is from the a different backend."""
        if self.backend.name == keras.backend.backend():
            return weight
        else:
            weight = keras.ops.convert_to_numpy(weight)
            return self.backend.convert_to_tensor(weight)
