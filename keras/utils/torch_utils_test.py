import numpy as np
import pytest
import torch

from keras import backend
from keras import layers
from keras import models
from keras import testing
from keras.utils.torch_utils import TorchModuleWrapper


class Classifier(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = TorchModuleWrapper(torch.nn.Linear(2, 4))

    def call(self, x):
        return self.fc(x)


class ClassifierWithNoSpecialCasing(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = torch.nn.Linear(2, 4)
        self.fc2 = layers.Dense(2)

    def call(self, x):
        return self.fc(self.fc2(x))


class TorchUtilsTest(testing.TestCase):
    @pytest.mark.skipif(
        backend.backend() != "torch", reason="Requires torch backend"
    )
    def test_basic_usage(self):
        model = Classifier()
        self.assertEqual(len(model.layers), 1)
        self.assertEqual(len(model.trainable_weights), 2)
        model(np.random.random((3, 2)))
        model.compile(optimizer="sgd", loss="mse")
        model.fit(np.random.random((3, 2)), np.random.random((3, 4)))

    @pytest.mark.skipif(
        backend.backend() != "torch", reason="Requires torch backend"
    )
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
        model.fit(np.random.random((3, 2)), np.random.random((3, 4)))
