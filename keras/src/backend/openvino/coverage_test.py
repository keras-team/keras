

import os
import numpy as np
import pytest
from keras.src import testing
from keras.src import backend

class OpenVINOCoverageTest(testing.TestCase):

    def setUp(self):
        super().setUp()
        if backend.backend() != "openvino":
            self.skipTest("OpenVINO backend not active")

    @property
    def nn(self):
        from keras.src.backend.openvino import nn
        return nn

    @property
    def image(self):
        from keras.src.backend.openvino import image
        return image

    def test_rgb_to_grayscale(self):
        # Shape: (2, 5, 5, 3) - NHWC
        x_np = np.random.random((2, 5, 5, 3)).astype("float32")
        x = backend.convert_to_tensor(x_np)
        # Standard weights: 0.2989 * R + 0.5870 * G + 0.1140 * B
        expected = (
            0.2989 * x_np[..., 0] + 
            0.5870 * x_np[..., 1] + 
            0.1140 * x_np[..., 2]
        )
        expected = np.expand_dims(expected, axis=-1)

        # Test default (channels_last)
        out = self.image.rgb_to_grayscale(x, data_format="channels_last")
        self.assertAllClose(out, expected, atol=1e-5)
        
        # Test channels_first
        x_cf = np.transpose(x_np, (0, 3, 1, 2))
        x_cf_t = backend.convert_to_tensor(x_cf)
        expected_cf = np.transpose(expected, (0, 3, 1, 2))
        out_cf = self.image.rgb_to_grayscale(x_cf_t, data_format="channels_first")
        self.assertEqual(out_cf.shape, (2, 1, 5, 5))
        self.assertAllClose(out_cf, expected_cf, atol=1e-5)

    def test_adaptive_average_pool(self):
        # Shape: (1, 6, 6, 1)
        x_np = np.arange(36).reshape((1, 6, 6, 1)).astype("float32")
        x = backend.convert_to_tensor(x_np)
        
        # Output pool size: (3, 3)
        # 6x6 -> 3x3 means 2x2 blocks
        out = self.nn.adaptive_average_pool(x, (3, 3), data_format="channels_last")
        
        # Top-left block: 0, 1, 6, 7 -> avg 3.5
        self.assertAllClose(out[0, 0, 0, 0], 3.5, atol=1e-5)

    def test_adaptive_max_pool(self):
        # Shape: (1, 6, 6, 1)
        x_np = np.arange(36).reshape((1, 6, 6, 1)).astype("float32")
        x = backend.convert_to_tensor(x_np)
        
        out = self.nn.adaptive_max_pool(x, (3, 3), data_format="channels_last")
        
        # Top-left block: 0, 1, 6, 7 -> max 7
        self.assertAllClose(out[0, 0, 0, 0], 7.0, atol=1e-5)

    def test_categorical_crossentropy(self):
        # target: (2, 3), output: (2, 3)
        target = np.array([[1., 0., 0.], [0., 1., 0.]], dtype="float32")
        output = np.array([[0.9, 0.05, 0.05], [0.05, 0.89, 0.06]], dtype="float32")
        
        # From logits=False
        # log(0.9) ~ -0.105, log(0.89) ~ -0.116
        out = self.nn.categorical_crossentropy(target, output, from_logits=False)
        # Expected: [-log(0.9), -log(0.89)]
        expected = -np.log([0.9, 0.89])
        self.assertAllClose(out, expected, atol=1e-4)

        # From logits=True
        # logits -> softmax -> log
        logits = np.array([[10., 0., 0.], [0., 10., 0.]], dtype="float32")
        # Softmax([10,0,0]) is approx [1, 0, 0]
        out_logits = self.nn.categorical_crossentropy(target, logits, from_logits=True)
        self.assertAllClose(out_logits, [0., 0.], atol=1e-3)

    def test_sparse_categorical_crossentropy(self):
        # target: (2,), output: (2, 3)
        target = np.array([0, 1], dtype="int32")
        output = np.array([[0.9, 0.05, 0.05], [0.05, 0.89, 0.06]], dtype="float32")
        
        out = self.nn.sparse_categorical_crossentropy(target, output, from_logits=False)
        expected = -np.log([0.9, 0.89])
        self.assertAllClose(out, expected, atol=1e-4)

    def test_binary_crossentropy(self):
        # target: (2, 1), output: (2, 1)
        target = np.array([[1.], [0.]], dtype="float32")
        output = np.array([[0.9], [0.1]], dtype="float32")
        
        # -[1*log(0.9) + 0, 0 + 1*log(1-0.1)] = [-log(0.9), -log(0.9)]
        out = self.nn.binary_crossentropy(target, output, from_logits=False)
        expected = -np.log([0.9, 0.9])
        self.assertAllClose(out, expected, atol=1e-4)
        
        # from_logits=True
        # sigmoid(2.197) ~= 0.9
        logits = np.array([[2.1972], [-2.1972]], dtype="float32") 
        out_logits = self.nn.binary_crossentropy(target, logits, from_logits=True)
        self.assertAllClose(out_logits, expected, atol=1e-2)

