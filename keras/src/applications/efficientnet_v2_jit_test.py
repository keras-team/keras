"""Test for Issue #21647: jit_compile=True with EfficientNetV2 on torch
backend."""


import numpy as np
import pytest

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.applications.efficientnet_v2 import EfficientNetV2B2
from keras.src.losses import CategoricalCrossentropy
from keras.src.optimizers import Adam


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="This test is specifically for torch backend",
)
class EfficientNetV2JitCompileTest(testing.TestCase):
    """Test EfficientNetV2 models with jit_compile=True on torch backend."""

    def test_efficientnet_v2_b2_with_jit_compile(self):
        """Test that EfficientNetV2B2 works with jit_compile=True."""
        num_classes = 10
        batch_size = 2  # Small batch for testing
        steps_per_epoch = 1
        epochs = 1

        # Generate random data (use minimum supported size)
        # Torch backend uses channels_first format: (C, H, W)
        data_shape = (3, 260, 260)  # Default size for EfficientNetV2B2
        x_train = np.random.rand(
            batch_size * steps_per_epoch, *data_shape
        ).astype(np.float32)
        y_train = np.random.randint(
            0, num_classes, size=(batch_size * steps_per_epoch,)
        )
        y_train = np.eye(num_classes)[y_train]

        # Create model
        base_model = EfficientNetV2B2(
            include_top=False,
            input_shape=(3, 260, 260),  # Fixed shape (channels_first)
            pooling="avg",
            include_preprocessing=True,
            weights=None,  # Don't load weights for faster testing
        )
        x = base_model.output
        output = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(inputs=base_model.input, outputs=output)

        # Compile with jit_compile=True
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossentropy(),
            metrics=["accuracy"],
            jit_compile=True,
        )

        # This should not raise InternalTorchDynamoError
        history = model.fit(
            x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0
        )

        # Basic sanity check
        self.assertIsNotNone(history)
        self.assertIn("loss", history.history)

    def test_efficientnet_v2_b0_with_jit_compile(self):
        """Test that EfficientNetV2B0 also works with jit_compile=True."""
        from keras.src.applications.efficientnet_v2 import EfficientNetV2B0

        num_classes = 5
        batch_size = 2

        # Generate random data
        # Torch backend uses channels_first format: (C, H, W)
        x_train = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
        _ = np.eye(num_classes)[
            np.random.randint(0, num_classes, size=(batch_size,))
        ]

        # Create model
        base_model = EfficientNetV2B0(
            include_top=False,
            input_shape=(3, 224, 224),  # channels_first format for torch
            pooling="avg",
            weights=None,
        )
        x = base_model.output
        output = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(inputs=base_model.input, outputs=output)

        # Compile with jit_compile=True
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossentropy(),
            metrics=["accuracy"],
            jit_compile=True,
        )

        # Should work without errors
        predictions = model.predict(x_train, verbose=0)
        self.assertEqual(predictions.shape, (batch_size, num_classes))


if __name__ == "__main__":
    pytest.main([__file__])
