import os

import numpy as np
import pytest
import torch
from absl.testing import parameterized

import keras
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import saving
from keras.src import testing
from keras.src.backend.torch.core import get_device
from keras.src.utils.torch_utils import TorchModuleWrapper


class Classifier(models.Model):
    def __init__(
        self, use_batch_norm=False, num_torch_layers=1, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_batch_norm = use_batch_norm
        self.num_torch_layers = num_torch_layers
        self.torch_wrappers = []
        for _ in range(num_torch_layers):
            modules = [torch.nn.Linear(2, 2)]
            if use_batch_norm:
                modules.append(torch.nn.BatchNorm1d(2))
            torch_model = torch.nn.Sequential(*modules)
            self.torch_wrappers.append(TorchModuleWrapper(torch_model))
        self.fc = layers.Dense(1)

    def call(self, x, training=None):
        for wrapper in self.torch_wrappers:
            x = wrapper(x, training=training)
        return self.fc(x)

    def get_config(self):
        config = super().get_config()
        config["use_batch_norm"] = self.use_batch_norm
        config["num_torch_layers"] = self.num_torch_layers
        return config


class ClassifierWithNoSpecialCasing(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = torch.nn.Linear(2, 4)
        self.bn1 = torch.nn.BatchNorm1d(4)
        self.fc2 = torch.nn.Linear(4, 4)
        self.fc3 = layers.Dense(2)

    def call(self, x, training=None):
        return self.fc3(self.fc2(self.bn1(self.fc1(x))))


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
class TorchUtilsTest(testing.TestCase):
    @parameterized.parameters(
        {"use_batch_norm": False, "num_torch_layers": 1},
        {"use_batch_norm": True, "num_torch_layers": 1},
    )
    def test_basic_usage(self, use_batch_norm, num_torch_layers):
        model = Classifier(use_batch_norm, num_torch_layers)
        self.assertEqual(len(model.layers), 2)
        # Linear - Weights, bias, BN - beta, gamma
        torch_trainable_count = 0
        for i, layer in zip(range(num_torch_layers), model.torch_wrappers):
            layer_trainable_count = 2
            if use_batch_norm:
                layer_trainable_count += 2
            self.assertEqual(
                len(layer.trainable_weights), layer_trainable_count
            )
            torch_trainable_count += layer_trainable_count
        model(np.random.random((3, 2)))
        self.assertEqual(len(model.layers), 2 * num_torch_layers)
        self.assertEqual(
            len(model.trainable_weights), torch_trainable_count + 2
        )
        model.compile(optimizer="sgd", loss="mse")
        model.fit(np.random.random((3, 2)), np.random.random((3, 1)))

    @parameterized.named_parameters(
        (
            "explicit_torch_wrapper",
            Classifier,
            {"use_batch_norm": True, "num_torch_layers": 1},
        ),
        ("implicit_torch_wrapper", ClassifierWithNoSpecialCasing, {}),
    )
    def test_training_args(self, cls, kwargs):
        model = cls(**kwargs)
        model(np.random.random((3, 2)), training=False)  # Eager call to build
        ref_weights = model.get_weights()
        ref_running_mean = backend.convert_to_numpy(
            model.torch_wrappers[0].module[-1].running_mean
            if cls is Classifier
            else model.bn1.module.running_mean
        )

        # Test training=False doesn't affect model weights
        model(np.random.random((3, 2)), training=False)
        weights = model.get_weights()
        for w, ref_w in zip(weights, ref_weights):
            self.assertAllClose(w, ref_w)

        # Test training=None affects BN's stats
        model.set_weights(ref_weights)  # Restore previous weights
        model(np.random.random((3, 2)))
        running_mean = backend.convert_to_numpy(
            model.torch_wrappers[0].module[-1].running_mean
            if cls is Classifier
            else model.bn1.module.running_mean
        )
        self.assertNotAllClose(running_mean, ref_running_mean)

        # Test training=True affects BN's stats
        model.set_weights(ref_weights)  # Restore previous weights
        model(np.random.random((3, 2)), training=True)
        running_mean = backend.convert_to_numpy(
            model.torch_wrappers[0].module[-1].running_mean
            if cls is Classifier
            else model.bn1.module.running_mean
        )
        self.assertNotAllClose(running_mean, ref_running_mean)

    def test_module_autowrapping(self):
        model = ClassifierWithNoSpecialCasing()
        self.assertIsInstance(model.fc1, TorchModuleWrapper)
        self.assertIsInstance(model.bn1, TorchModuleWrapper)
        self.assertIsInstance(model.fc2, TorchModuleWrapper)
        self.assertFalse(isinstance(model.fc3, TorchModuleWrapper))
        self.assertEqual(len(model.fc1.trainable_weights), 2)
        self.assertEqual(len(model.bn1.trainable_weights), 2)
        self.assertEqual(len(model.fc2.trainable_weights), 2)
        model(np.random.random((3, 2)))
        self.assertEqual(len(model.layers), 4)
        self.assertEqual(len(model.fc3.trainable_weights), 2)
        self.assertEqual(len(model.trainable_weights), 8)
        model.compile(optimizer="sgd", loss="mse")
        model.fit(np.random.random((3, 2)), np.random.random((3, 2)))

    def test_load_weights_autowrapping(self):
        # Test loading weights
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.weights.h5")
        model = ClassifierWithNoSpecialCasing()
        model.compile(optimizer="sgd", loss="mse")
        x, y = np.random.random((3, 2)), np.random.random((3, 1))
        x_test, y_test = np.random.random((3, 2)), np.random.random((3, 1))
        model.fit(x, y)
        ref_loss = model.evaluate(x_test, y_test)
        model.save_weights(temp_filepath)

        new_model = ClassifierWithNoSpecialCasing()
        new_model(np.random.random((3, 2)))
        new_model.compile(optimizer="sgd", loss="mse")
        new_model.load_weights(temp_filepath)
        for ref_w, new_w in zip(model.get_weights(), new_model.get_weights()):
            self.assertAllClose(ref_w, new_w, atol=1e-5)
        loss = new_model.evaluate(x_test, y_test)
        self.assertAllClose(ref_loss, loss, atol=1e-5)

    def test_serialize_model_autowrapping(self):
        # Test loading saved model
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        model = ClassifierWithNoSpecialCasing()
        model.compile(optimizer="sgd", loss="mse")
        x, y = np.random.random((3, 2)), np.random.random((3, 1))
        x_test, y_test = np.random.random((3, 2)), np.random.random((3, 1))
        model.fit(x, y)
        ref_loss = model.evaluate(x_test, y_test)
        model.save(temp_filepath)

        new_model = saving.load_model(temp_filepath)
        for ref_w, new_w in zip(model.get_weights(), new_model.get_weights()):
            self.assertAllClose(ref_w, new_w, atol=1e-5)
        loss = new_model.evaluate(x_test, y_test)
        self.assertAllClose(ref_loss, loss, atol=1e-5)

    @parameterized.parameters(
        {"use_batch_norm": False, "num_torch_layers": 1},
        {"use_batch_norm": True, "num_torch_layers": 1},
        {"use_batch_norm": False, "num_torch_layers": 2},
        {"use_batch_norm": True, "num_torch_layers": 2},
    )
    def test_load_weights(self, use_batch_norm, num_torch_layers):
        # Test loading weights
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.weights.h5")
        model = Classifier(use_batch_norm, num_torch_layers)
        model.compile(optimizer="sgd", loss="mse")
        x, y = np.random.random((3, 2)), np.random.random((3, 1))
        x_test, y_test = np.random.random((3, 2)), np.random.random((3, 1))
        model.fit(x, y)
        ref_loss = model.evaluate(x_test, y_test)
        model.save_weights(temp_filepath)

        new_model = Classifier(use_batch_norm, num_torch_layers)
        new_model(np.random.random((3, 2)))
        new_model.compile(optimizer="sgd", loss="mse")
        new_model.load_weights(temp_filepath)
        for ref_w, new_w in zip(model.get_weights(), new_model.get_weights()):
            self.assertAllClose(ref_w, new_w, atol=1e-5)
        loss = new_model.evaluate(x_test, y_test)
        self.assertAllClose(ref_loss, loss, atol=1e-5)

    @parameterized.parameters(
        {"use_batch_norm": False, "num_torch_layers": 1},
        {"use_batch_norm": True, "num_torch_layers": 1},
        {"use_batch_norm": False, "num_torch_layers": 2},
        {"use_batch_norm": True, "num_torch_layers": 2},
    )
    def test_serialize_model(self, use_batch_norm, num_torch_layers):
        # Test loading saved model
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        model = Classifier(use_batch_norm, num_torch_layers)
        model.compile(optimizer="sgd", loss="mse")
        x, y = np.random.random((3, 2)), np.random.random((3, 1))
        x_test, y_test = np.random.random((3, 2)), np.random.random((3, 1))
        model.fit(x, y)
        ref_loss = model.evaluate(x_test, y_test)
        model.save(temp_filepath)

        new_model = saving.load_model(temp_filepath)
        for ref_w, new_w in zip(model.get_weights(), new_model.get_weights()):
            self.assertAllClose(ref_w, new_w, atol=1e-5)
        loss = new_model.evaluate(x_test, y_test)
        self.assertAllClose(ref_loss, loss, atol=1e-5)

    def test_from_config(self):
        module = torch.nn.Sequential(torch.nn.Linear(2, 4))
        mw = TorchModuleWrapper(module)
        config = mw.get_config()
        new_mw = TorchModuleWrapper.from_config(config)
        for ref_w, new_w in zip(mw.get_weights(), new_mw.get_weights()):
            self.assertAllClose(ref_w, new_w, atol=1e-5)

    def test_build_model(self):
        x = keras.Input([4])
        z = TorchModuleWrapper(torch.nn.Linear(4, 8), output_shape=[None, 8])(x)
        y = TorchModuleWrapper(torch.nn.Linear(8, 16), output_shape=[None, 16])(
            z
        )
        model = keras.Model(x, y)
        self.assertEqual(model.predict(np.zeros([5, 4])).shape, (5, 16))
        self.assertEqual(model(np.zeros([5, 4])).shape, (5, 16))

    @parameterized.named_parameters(
        ("safe_mode", True),
        ("unsafe_mode", False),
    )
    def test_save_load(self, safe_mode):
        @keras.saving.register_keras_serializable()
        class M(keras.Model):
            def __init__(self, module, **kwargs):
                super().__init__(**kwargs)
                self.module = module

            def call(self, x):
                return self.module(x)

            def get_config(self):
                base_config = super().get_config()
                config = {"module": self.module}
                return {**base_config, **config}

            @classmethod
            def from_config(cls, config):
                config["module"] = saving.deserialize_keras_object(
                    config["module"]
                )
                return cls(**config)

        m = M(torch.nn.Conv2d(1, 10, kernel_size=(3, 3)))
        device = get_device()  # Get the current device (e.g., "cuda" or "cpu")
        x = torch.ones(
            (10, 1, 28, 28), device=device
        )  # Place input on the correct device
        ref_output = m(x)
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        m.save(temp_filepath)

        if safe_mode:
            with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
                saving.load_model(temp_filepath, safe_mode=safe_mode)
        else:
            new_model = saving.load_model(temp_filepath, safe_mode=safe_mode)
            self.assertAllClose(new_model(x), ref_output)
