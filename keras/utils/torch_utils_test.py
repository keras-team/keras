import os

import numpy as np
import pytest
import torch
from absl.testing import parameterized

from keras import backend
from keras import layers
from keras import models
from keras import saving
from keras import testing
from keras.utils.torch_utils import TorchModuleWrapper


class Classifier(models.Model):
    def __init__(self, use_batch_norm=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_batch_norm = use_batch_norm
        modules = [torch.nn.Linear(2, 4)]
        if use_batch_norm:
            modules.append(torch.nn.BatchNorm1d(4))
        torch_model = torch.nn.Sequential(*modules)
        self.fc = TorchModuleWrapper(torch_model)
        self.fc2 = layers.Dense(1)

    def call(self, x):
        return self.fc2(self.fc(x))

    def get_config(self):
        config = super().get_config()
        config["use_batch_norm"] = self.use_batch_norm
        return config


class ClassifierWithNoSpecialCasing(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = torch.nn.Linear(2, 4)
        self.fc2 = layers.Dense(2)

    def call(self, x):
        return self.fc2(self.fc(x))


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
class TorchUtilsTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        {"use_batch_norm": False},
        {"use_batch_norm": True},
    )
    def test_basic_usage(self, use_batch_norm):
        model = Classifier(use_batch_norm)
        self.assertEqual(len(model.layers), 2)
        # Linear - Weights, bias, BN - beta, gamma
        fc_trainable_count = 2
        if use_batch_norm:
            fc_trainable_count += 2
            self.assertEqual(
                len(model.fc.trainable_weights), fc_trainable_count
            )
        model(np.random.random((3, 2)))
        self.assertEqual(len(model._layers), 2)
        self.assertEqual(len(model.trainable_weights), fc_trainable_count + 2)
        model.compile(optimizer="sgd", loss="mse")
        model.fit(np.random.random((3, 2)), np.random.random((3, 1)))

    def test_module_autowrapping(self):
        model = ClassifierWithNoSpecialCasing()
        self.assertIsInstance(model.fc, TorchModuleWrapper)
        self.assertFalse(isinstance(model.fc2, TorchModuleWrapper))
        self.assertEqual(len(model.fc.trainable_weights), 2)
        model(np.random.random((3, 2)))
        self.assertEqual(len(model._layers), 2)
        self.assertEqual(len(model.fc2.trainable_weights), 2)
        self.assertEqual(len(model.trainable_weights), 4)
        model.compile(optimizer="sgd", loss="mse")
        model.fit(np.random.random((3, 2)), np.random.random((3, 2)))

    @parameterized.parameters(
        {"use_batch_norm": False},
        {"use_batch_norm": True},
    )
    def test_load_weights(self, use_batch_norm):
        # Test loading weights
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.weights.h5")
        model = Classifier(use_batch_norm)
        model.compile(optimizer="sgd", loss="mse")
        x, y = np.random.random((3, 2)), np.random.random((3, 1))
        x_test, y_test = np.random.random((3, 2)), np.random.random((3, 1))
        model.fit(x, y)
        ref_loss = model.evaluate(x_test, y_test)
        model.save_weights(temp_filepath)

        new_model = Classifier(use_batch_norm)
        new_model(np.random.random((3, 2)))
        new_model.compile(optimizer="sgd", loss="mse")
        new_model.load_weights(temp_filepath)
        for ref_w, new_w in zip(model.get_weights(), new_model.get_weights()):
            self.assertAllClose(ref_w, new_w, atol=1e-5)
        loss = new_model.evaluate(x_test, y_test)
        self.assertAllClose(ref_loss, loss, atol=1e-5)

    @parameterized.parameters(
        {"use_batch_norm": False},
        {"use_batch_norm": True},
    )
    def test_serialize_model(self, use_batch_norm):
        # Test loading saved model
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        model = Classifier(use_batch_norm)
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
