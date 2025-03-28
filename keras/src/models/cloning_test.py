import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import initializers
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src import tree
from keras.src.models.cloning import clone_model
from keras.src.models.functional import Functional


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
            config["name"] = config["name"] + "_custom"
            return layer.__class__.from_config(config)

        model = model_fn()
        new_model = clone_model(model, clone_function=clone_function)
        for l1, l2 in zip(model.layers, new_model.layers):
            if not isinstance(l1, layers.InputLayer):
                self.assertEqual(l2.name, l1.name + "_custom")

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

    def test_call_fn_custom_layer_replace(self):
        # alternative dense implementation using the same weights
        class AltDense(layers.Layer):
            def __init__(self, layer, **kwargs):
                super().__init__(**kwargs)
                self.dense_layer = layer

            def build(self, input_shape):
                self.w = self.dense_layer.kernel
                self.b = self.dense_layer.bias

            def call(self, inputs):
                result = ops.matmul(inputs, self.w) + self.b
                return result

        model = get_mlp_functional_model(shared_layers=False)

        def call_function(layer, *args, **kwargs):
            if isinstance(layer, layers.Dense):
                new_layer = AltDense(layer)
                return new_layer(*args, **kwargs)
            else:
                return layer(*args, **kwargs)

        new_model = clone_model(
            model,
            clone_function=lambda x: x,  # everything happense in call_function.
            call_function=call_function,
        )
        self.assertLen(model.layers, 3)
        self.assertLen(new_model.layers, 3)
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

    def test_func_subclass(self):
        const_init = initializers.Ones()
        zero_init = initializers.Zeros()

        # alternative dense implementation
        class AltDense(layers.Layer):
            def __init__(self, units, **kwargs):
                super().__init__(**kwargs)
                self.units = units

            def build(self, input_shape):
                self.w = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer=const_init,
                    trainable=True,
                )
                self.b = self.add_weight(
                    shape=(self.units,),
                    initializer=zero_init,
                    trainable=True,
                )

            def call(self, inputs):
                return ops.matmul(inputs, self.w) + self.b

        class FuncSubclassModel(models.Model):
            def __init__(self, **kwargs):
                inputs = layers.Input(shape=(8,))
                y = layers.Dense(
                    4, kernel_initializer=const_init, name="original1"
                )(inputs)
                outputs = layers.Dense(
                    8, kernel_initializer=const_init, name="original2"
                )(y)
                super().__init__(inputs, outputs, **kwargs)

        inputs = layers.Input(shape=(12,))
        y = layers.Dense(8, kernel_initializer=const_init, name="original3")(
            inputs
        )
        funcsub = FuncSubclassModel()
        y = funcsub(y)
        outputs = funcsub(y)  # reused layer
        model = models.Model(inputs, outputs)

        data = np.random.uniform(size=(4, 12))
        model(data)

        def replace_fn(layer, *args, **kwargs):
            if isinstance(layer, layers.Dense):
                return AltDense(layer.units)(*args, **kwargs)
            else:
                return layer(*args, **kwargs)  # pass-through

        model2 = clone_model(
            model,
            input_tensors=inputs,
            # everything happense in call_function.
            clone_function=lambda x: x,
            call_function=replace_fn,
            recursive=True,
        )

        model2(data)

        # original model is unchanged
        for variable in model.variables:
            self.assertContainsSubsequence(variable.path, "original")

        # new model has new AltDense layers
        for variable in model2.variables:
            self.assertContainsSubsequence(variable.path, "alt_dense")

        self.assertEqual(len(model.layers), len(model2.layers))
        for layer1, layer2 in zip(model.layers, model2.layers):
            if isinstance(layer1, layers.Dense):
                self.assertTrue(layer2.__class__ is AltDense)
            # A subclass of Functional is cloned as vanilla Functional for now
            # unless it has an explicit functional constructor
            elif isinstance(layer1, FuncSubclassModel):
                self.assertTrue(
                    layer2.__class__ is Functional
                    or layer2.__class__ is FuncSubclassModel
                )
            else:
                self.assertEqual(layer1.__class__, layer2.__class__)

        self.assertAllClose(model(data), model2(data))

    def test_parametrized_func_subclass(self):
        # alternative dense implementation
        class AltDense(layers.Layer):
            def __init__(self, units, **kwargs):
                super().__init__(**kwargs)
                self.units = units

            def build(self, input_shape):
                self.w = self.add_weight(shape=(input_shape[-1], self.units))
                self.b = self.add_weight(shape=(self.units,))

            def call(self, inputs):
                return ops.matmul(inputs, self.w) + self.b

        class FuncSubclassParametrizedModel(models.Model):
            def __init__(self, *args, param=4, **kwargs):
                inputs = layers.Input(shape=(8,))
                y = layers.Dense(param)(inputs)
                outputs = layers.Dense(param)(y)
                super().__init__(inputs, outputs, *args, **kwargs)
                self.param = param

        def replace_fn(layer, *args, **kwargs):
            if isinstance(layer, layers.Dense):
                return AltDense(layer.units)(*args, **kwargs)
            else:
                return layer(*args, **kwargs)  # pass-through

        model = FuncSubclassParametrizedModel(param=11)
        self.assertEqual(model.param, 11)

        model2 = clone_model(
            model,
            clone_function=lambda x: x,
            call_function=replace_fn,
            recursive=True,
        )
        # A subclass of Functional is cloned as vanilla Functional for now
        self.assertFalse(model2.__class__ == FuncSubclassParametrizedModel)
        self.assertTrue(model2.__class__ == Functional)
        # test that the layers were replaced
        self.assertTrue(isinstance(model2.layers[0], layers.InputLayer))
        self.assertTrue(isinstance(model2.layers[1], AltDense))
        # Even though the cloned FuncSubclassParametrizedModel is now
        # a valilla Functional, test that the underlying AltDense layers
        # have the correct param size, as set by the param value.
        self.assertEqual(model2.layers[1].w.shape[1], 11)

    def test_clone_passthrough_subfunctional(self):
        class SubFunctional(models.Model):
            pass

        inputs = layers.Input(shape=(8,))
        y = layers.Dense(4)(inputs)
        outputs = layers.Dense(8)(y)
        model = SubFunctional(inputs, outputs)

        model2 = clone_model(model)
        # cloned as a vanilla Functional
        self.assertTrue(model2.__class__ == Functional)

    def test_clone_passthrough_subfunctional_recursive(self):
        class SubFunctional(models.Model):
            pass

        inputs = layers.Input(shape=(8,))
        outputs = layers.Dense(8)(inputs)
        sublayer = SubFunctional(inputs, outputs)

        inputs = layers.Input(shape=(8,))
        outputs = sublayer(inputs)
        model = models.Model(inputs, outputs)

        model2 = clone_model(model, recursive=True)
        # cloned as a vanilla Functional
        self.assertTrue(model2.__class__ == Functional)
        self.assertTrue(model2.layers[1].__class__ == Functional)

    def test_clone_functional_subclass(self):
        class SubFunctional(models.Model):
            def __init__(self, *args, **kwargs):
                inputs = layers.Input(shape=(8,))
                outputs = layers.Dense(8)(inputs)
                return super().__init__(inputs, outputs, *args, **kwargs)

        model = SubFunctional()

        model2 = clone_model(model)
        # cloned as a vanilla Functional
        self.assertTrue(model2.__class__ == Functional)

    def test_clone_functional_subclass_non_recursive(self):
        class SubFunctional(models.Model):
            def __init__(self, *args, **kwargs):
                inputs = layers.Input(shape=(8,))
                outputs = layers.Dense(4)(inputs)
                return super().__init__(inputs, outputs, *args, **kwargs)

        inputs = layers.Input(shape=(8,))
        outputs = SubFunctional()(inputs)
        model = models.Model(inputs, outputs)

        model2 = clone_model(model)
        self.assertTrue(model2.__class__ == Functional)
        # not touched in non-recursive mode
        self.assertTrue(model2.layers[1].__class__ == SubFunctional)

    def test_clone_functional_subclass_recursive(self):
        class SubFunctional(models.Model):
            def __init__(self, *args, **kwargs):
                inputs = layers.Input(shape=(8,))
                outputs = layers.Dense(4)(inputs)
                return super().__init__(inputs, outputs, *args, **kwargs)

        inputs = layers.Input(shape=(8,))
        outputs = SubFunctional()(inputs)
        model = models.Model(inputs, outputs)

        model2 = clone_model(model, clone_function=lambda x: x, recursive=True)
        self.assertTrue(model2.__class__ == Functional)
        # cloned as a vanilla Functional
        self.assertTrue(model2.layers[1].__class__ == Functional)

    def test_clone_functional_subclass_non_recursive2(self):
        class SubFunctional(models.Model):
            def __init__(self, *args, **kwargs):
                inputs = layers.Input(shape=(8,))
                outputs = layers.Dense(4)(inputs)
                return super().__init__(inputs, outputs, *args, **kwargs)

        inputs = layers.Input(shape=(8,))
        outputs = SubFunctional()(inputs)
        model = models.Model(inputs, outputs)

        model2 = clone_model(model, clone_function=lambda x: x, recursive=False)
        self.assertTrue(model2.__class__ == Functional)
        # not touched in non-recursive mode
        self.assertTrue(model2.layers[1].__class__ == SubFunctional)

    def test_clone_passthrough_subfunctional_with_params(self):
        class SubFunctional(models.Model):
            def __init__(self, inputs, outputs, param, *args, **kwargs):
                super().__init__(inputs, outputs, *args, **kwargs)
                self.param = param

        inputs = layers.Input(shape=(8,))
        y = layers.Dense(4)(inputs)
        outputs = layers.Dense(8)(y)
        model = SubFunctional(inputs, outputs, 8)

        # cloned as a vanilla Functional
        model2 = clone_model(model)
        self.assertTrue(model2.__class__ == Functional)
