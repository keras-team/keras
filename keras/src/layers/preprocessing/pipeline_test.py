import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class CanaryLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        self.training = None
        self.received_mask = False

    def call(self, x, training=False, mask=None):
        self.training = training
        if mask is not None:
            self.received_mask = True
        return x

    def compute_mask(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class PipelineTest(testing.TestCase):
    def test_basics(self):
        run_training_check = False if backend.backend() == "numpy" else True
        self.run_layer_test(
            layers.Pipeline,
            init_kwargs={
                "layers": [layers.AutoContrast(), layers.RandomBrightness(0.1)],
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
            run_mixed_precision_check=False,
            run_training_check=run_training_check,
        )

    @pytest.mark.skipif(
        backend.backend() == "numpy", reason="masking not working in numpy"
    )
    def test_correctness(self):
        pipeline = layers.Pipeline([CanaryLayer(), CanaryLayer()])
        x = np.array([0])
        mask = np.array([0])
        pipeline(x, training=True, mask=mask)
        self.assertTrue(pipeline.layers[0].training)
        self.assertTrue(pipeline.layers[0].received_mask)
        self.assertTrue(pipeline.layers[1].training)
        self.assertTrue(pipeline.layers[1].received_mask)

    def test_tf_data_compatibility(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 12, 3)
            output_shape = (2, 8, 9, 3)
        else:
            input_shape = (2, 3, 10, 12)
            output_shape = (2, 3, 8, 9)
        layer = layers.Pipeline(
            [
                layers.AutoContrast(),
                layers.CenterCrop(8, 9),
            ]
        )
        input_data = np.random.random(input_shape)
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertEqual(tuple(output.shape), output_shape)

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Fails on CI, passes locally. TODO: debug",
    )
    def test_from_config(self):
        pipeline = layers.Pipeline(
            [
                layers.AutoContrast(),
                layers.CenterCrop(8, 9),
            ]
        )
        x = np.ones((2, 10, 12, 3))
        output = pipeline(x)
        restored = layers.Pipeline.from_config(pipeline.get_config())
        restored_output = restored(x)
        self.assertEqual(tuple(output.shape), (2, 8, 9, 3))
        self.assertAllClose(output, restored_output)
