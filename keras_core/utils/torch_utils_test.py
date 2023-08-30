import numpy as np
import torch

from keras_core import models
from keras_core import testing
from keras_core.utils.torch_utils import TorchModuleWrapper


class Classifier(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = TorchModuleWrapper(torch.nn.Linear(2, 4))

    def call(self, x):
        return self.fc(x)


class TorchUtilsTest(testing.TestCase):
    def test_basic_usage(self):
        model = Classifier()
        self.assertEqual(len(model.layers), 1)
        self.assertEqual(len(model.trainable_weights), 2)
        model(np.random.random((3, 2)))
        model.compile(optimizer="sgd", loss="mse")
        model.fit(np.random.random((3, 2)), np.random.random((3, 4)))
