import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src import tree
from keras.src.models.cloning import clone_model


def get_mlp_functional_model(shared_layers=False):
    inputs = layers.Input(shape=(3,))
    x = layers.Dense(2)(inputs)
    if shared_layers:
        layer = layers.Dense(2, name="shared")
        x = layer(x)
        x = layer(x)
    outputs = layers.Dense(2)(x)
    model = models.Model(inputs, outputs)
    return model


def get_nested_functional_model():
    inputs = layers.Input(shape=(4,))
    x = layers.Dense(3)(inputs)
    mlp = get_mlp_functional_model()
    x = mlp(x)
    outputs = layers.Dense(2)(x)
    model = models.Model(inputs, outputs)
    return model


def get_nested_sequential_model():
    model = models.Sequential()
    model.add(layers.Dense(2))
    model.add(get_sequential_model(explicit_input=False))
    model.add(layers.Dense(2))
    return model


def get_cnn_functional_model(shared_layers=False):
    inputs = layers.Input(shape=(7, 3))
    x = layers.Conv1D(2, 2, padding="same")(inputs)
    if shared_layers:
        layer = layers.Conv1D(2, 2, padding="same", name="shared")
        x = layer(x)
        x = layer(x)
    outputs = layers.Conv1D(2, 2, padding="same")(x)
    model = models.Model(inputs, outputs)
    return model


def get_sequential_model(explicit_input=True):
    model = models.Sequential()
    if explicit_input:
        model.add(layers.Input(shape=(3,)))
    model.add(layers.Dense(2))
    model.add(layers.Dense(2))
    return model


def get_cnn_sequential_model(explicit_input=True):
    model = models.Sequential()
    if explicit_input:
        model.add(layers.Input(shape=(7, 3)))
    model.add(layers.Conv1D(2, 2, padding="same"))
    model.add(layers.Conv1D(2, 2, padding="same"))
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
class CloneModelTest(testing.TestCase):
    def assert_models_equal(self, model1, model2, ref_input):
        result1 = model1(ref_input)
        result2 = model2(ref_input)
        for r1, r2 in zip(tree.flatten(result1), tree.flatten(result2)):
            self.assertAllClose(
                ops.convert_to_numpy(r1), ops.convert_to_numpy(r2)
            )

    def assert_weights_equal(self, model1, model2):
        for a, b in zip(model1.weights, model2.weights):
            self.assertAllClose(a.numpy(), b.numpy())

    @parameterized.named_parameters(
        ("mlp_functional", get_mlp_functional_model),
        ("cnn_functional", get_cnn_functional_model, True),
        ("sequential", get_sequential_model),
        (
            "deferred_sequential",
            lambda: get_sequential_model(explicit_input=False),
        ),
        ("subclassed", get_subclassed_model),
    )
    def test_cloning_correctness(self, model_fn, is_conv=False):
        ref_input = np.random.random((2, 7, 3) if is_conv else (2, 3))
        model = model_fn()
        new_model = clone_model(model)
        model(ref_input)  # Maybe needed to build the model
        new_model(ref_input)  # Maybe needed to build the model
        new_model.set_weights(model.get_weights())
        self.assert_models_equal(model, new_model, ref_input)

    @parameterized.named_parameters(
        ("mlp_functional", get_mlp_functional_model),
        ("cnn_functional", get_cnn_functional_model),
        ("sequential", get_sequential_model),
    )
    def test_custom_clone_function(self, model_fn):
        def clone_function(layer):
            config = layer.get_config()
            config["name"] = f"{config['name']}_custom"
            return layer.__class__.from_config(config)

        model = model_fn()
        new_model = clone_model(model, clone_function=clone_function)
        for l1, l2 in zip(model.layers, new_model.layers):
            if not isinstance(l1, layers.InputLayer):
                self.assertEqual(l2.name, f"{l1.name}_custom")

    @parameterized.named_parameters(
        ("cnn_functional", get_cnn_functional_model),
        ("cnn_sequential", get_cnn_sequential_model),
        (
            "cnn_sequential_noinputlayer",
            lambda: get_cnn_sequential_model(explicit_input=False),
        ),
    )
    def test_input_tensors(self, model_fn):
        ref_input = np.random.random((2, 7, 3))
        model = model_fn()
        model(ref_input)  # Maybe needed to get model inputs if no Input layer
        input_tensor = model.inputs[0]
        new_model = clone_model(model, input_tensors=input_tensor)
        tree.assert_same_structure(model.inputs, new_model.inputs)
        tree.assert_same_structure(model.outputs, new_model.outputs)

    def test_shared_layers_cloning(self):
        model = get_mlp_functional_model(shared_layers=True)
        new_model = clone_model(model)
        self.assertLen(new_model.layers, 4)

    def test_structured_io_cloning(self):
        x = layers.Input((3,))
        y = layers.Input((3,))
        z1 = x + y
        z2 = layers.Dense(5)(z1)
        inputs = dict(x=x, y=y)
        outputs = dict(z1=z1, z2=z2)
        model0 = models.Model(inputs, outputs)

        model = clone_model(model0)
        tree.assert_same_structure(model.input, inputs)
        tree.assert_same_structure(model.output, outputs)

        model = clone_model(model0, input_tensors=inputs)
        tree.assert_same_structure(model.input, inputs)
        tree.assert_same_structure(model.output, outputs)

        with self.assertRaisesRegex(
            ValueError,
            "`input_tensors` must have the same structure as model.input",
        ):
            model = clone_model(model0, input_tensors=(x, y))

    def test_call_fn(self):
        model = get_mlp_functional_model(shared_layers=False)

        def call_function(layer, *args, **kwargs):
            out = layer(*args, **kwargs)
            if isinstance(layer, layers.Dense):
                out = layers.Dropout(0.5)(out)
            return out

        new_model = clone_model(
            model,
            clone_function=lambda x: x,  # Reuse the same layers.
            call_function=call_function,
        )
        self.assertLen(model.layers, 3)
        self.assertLen(new_model.layers, 5)
        self.assertIsInstance(new_model.layers[2], layers.Dropout)
        self.assertIsInstance(new_model.layers[4], layers.Dropout)
        ref_input = np.random.random((2, 3))
        self.assert_models_equal(model, new_model, ref_input)

    def test_recursive(self):
        model = get_nested_functional_model()

        def call_function(layer, *args, **kwargs):
            out = layer(*args, **kwargs)
            if isinstance(layer, layers.Dense):
                out = layers.Dropout(0.5)(out)
            return out

        new_model = clone_model(
            model,
            clone_function=lambda x: x,  # Reuse the same layers.
            call_function=call_function,
            recursive=True,
        )
        self.assertLen(model._flatten_layers(), 8)
        self.assertLen(new_model._flatten_layers(), 12)
        self.assertIsInstance(new_model.layers[3].layers[2], layers.Dropout)
        self.assertIsInstance(new_model.layers[3].layers[4], layers.Dropout)
        ref_input = np.random.random((2, 4))
        self.assert_models_equal(model, new_model, ref_input)

        # Sequential.
        def clone_function(layer):
            layer = layer.__class__.from_config(layer.get_config())
            layer.flag = True
            return layer

        model = get_nested_sequential_model()
        new_model = clone_model(
            model,
            clone_function=clone_function,
            recursive=True,
        )
        ref_input = np.random.random((2, 3))
        model(ref_input)  # Maybe needed to build the model
        new_model(ref_input)  # Maybe needed to build the model
        new_model.set_weights(model.get_weights())
        self.assert_models_equal(model, new_model, ref_input)
        for l1, l2 in zip(model._flatten_layers(), new_model._flatten_layers()):
            if isinstance(l2, layers.Dense):
                self.assertFalse(hasattr(l1, "flag"))
                self.assertTrue(hasattr(l2, "flag"))

    def test_compiled_model_cloning(self):
        model = models.Sequential()
        model.add(layers.Input((3,)))
        model.add(layers.Dense(5, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy")
        cloned_model = clone_model(model)
        self.assertEqual(model.compiled, cloned_model.compiled)
