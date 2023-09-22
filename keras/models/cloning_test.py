import numpy as np
import pytest
from absl.testing import parameterized

from keras import layers
from keras import models
from keras import testing
from keras.models.cloning import clone_model


def get_functional_model(shared_layers=False):
    inputs = layers.Input(shape=(3,))
    x = layers.Dense(2)(inputs)
    if shared_layers:
        layer = layers.Dense(2, name="shared")
        x = layer(x)
        x = layer(x)
    outputs = layers.Dense(2)(x)
    model = models.Model(inputs, outputs)
    return model


def get_sequential_model(explicit_input=True):
    model = models.Sequential()
    if explicit_input:
        model.add(layers.Input(shape=(3,)))
    model.add(layers.Dense(2))
    model.add(layers.Dense(2))
    return model


def get_subclassed_model():
    class ExampleModel(models.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.d1 = layers.Dense(2)
            self.d2 = layers.Dense(2)

        def call(self, x):
            return self.d2(self.d1(x))

    return ExampleModel()


@pytest.mark.requires_trainable_backend
class CloneModelTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("functional", get_functional_model),
        ("sequential", get_sequential_model),
        (
            "deferred_sequential",
            lambda: get_sequential_model(explicit_input=False),
        ),
        ("subclassed", get_subclassed_model),
    )
    def test_cloning_correctness(self, model_fn):
        ref_input = np.random.random((2, 3))
        model = model_fn()
        new_model = clone_model(model)
        ref_output = model(ref_input)
        new_model(ref_input)  # Maybe needed to build the model
        new_model.set_weights(model.get_weights())
        output = new_model(ref_input)
        self.assertAllClose(ref_output, output)

    @parameterized.named_parameters(
        ("functional", get_functional_model),
        ("sequential", get_sequential_model),
    )
    def test_custom_clone_function(self, model_fn):
        def clone_function(layer):
            config = layer.get_config()
            config["name"] = config["name"] + "_custom"
            return layer.__class__.from_config(config)

        model = model_fn()
        new_model = clone_model(model, clone_function=clone_function)
        for l1, l2 in zip(model.layers, new_model.layers):
            if not isinstance(l1, layers.InputLayer):
                self.assertEqual(l2.name, l1.name + "_custom")

    def test_shared_layers_cloning(self):
        model = get_functional_model(shared_layers=True)
        new_model = clone_model(model)
        self.assertLen(new_model.layers, 4)
