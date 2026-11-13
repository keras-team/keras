from keras.src import ops
from keras.src import testing
from keras.src.backend.common import KerasTensor
from keras.src.backend.common.functional_scope import FunctionalScope
from keras.src.layers.core.input_layer import Input
from keras.src.models import Functional
from keras.src.models.model import model_from_json


class TestFunctionalScope(testing.TestCase):
    def verify_model(self, model, ref_inputs, ref_outputs):
        model_x = model(ref_inputs)
        self.assertAllClose(model_x, ref_outputs)

        json = model.to_json()

        reloaded_model = model_from_json(json)
        outputs = reloaded_model(ref_inputs)
        self.assertAllClose(outputs, ref_outputs)

        self.assertEqual(reloaded_model.to_json(), json)

    def test_no_input_op(self):
        with FunctionalScope():
            scope_x = ops.ones((2, 3), dtype="int32")

        eager_x = ops.ones((2, 3), dtype="int32")

        self.assertIsInstance(scope_x, KerasTensor)
        self.assertEqual(scope_x.shape, (2, 3))
        self.assertEqual(scope_x.dtype, "int32")

        self.assertNotIsInstance(eager_x, KerasTensor)
        self.assertTrue(ops.is_tensor(eager_x))

        inputs = Input(batch_shape=(2, 3), dtype="int32")
        outputs = ops.add(scope_x, inputs)
        model = Functional(inputs=inputs, outputs=outputs)

        self.verify_model(model, ops.zeros((2, 3), dtype="int32"), eager_x)

    def test_one_input_op(self):
        with FunctionalScope():
            scope_x = ops.full((2, 3), 4, dtype="int32")

        full_inputs = Input(batch_shape=(), dtype="int32")
        func_x = ops.full((2, 3), full_inputs, dtype="int32")

        eager_x = ops.full((2, 3), 4, dtype="int32")

        self.assertIsInstance(func_x, KerasTensor)
        self.assertEqual(func_x.shape, (2, 3))
        self.assertEqual(func_x.dtype, "int32")

        self.assertIsInstance(scope_x, KerasTensor)
        self.assertEqual(scope_x.shape, (2, 3))
        self.assertEqual(scope_x.dtype, "int32")

        self.assertNotIsInstance(eager_x, KerasTensor)
        self.assertTrue(ops.is_tensor(eager_x))

        inputs = Input(batch_shape=(2, 3), dtype="int32")
        outputs = ops.add(scope_x, inputs)
        model = Functional(inputs=inputs, outputs=outputs)

        self.verify_model(model, ops.zeros((2, 3), dtype="int32"), eager_x)

    def test_two_input_op(self):
        with FunctionalScope():
            scope_x = ops.add(1, 3)

        left_input = Input(batch_shape=(), dtype="int32")
        func_left_x = ops.add(left_input, 3)

        right_input = Input(batch_shape=(), dtype="int32")
        func_right_x = ops.add(1, right_input)

        eager_x = ops.add(1, 3)

        self.assertIsInstance(scope_x, KerasTensor)
        self.assertEqual(scope_x.shape, ())
        self.assertEqual(scope_x.dtype, "int32")

        self.assertIsInstance(func_left_x, KerasTensor)
        self.assertEqual(func_left_x.shape, ())
        self.assertEqual(func_left_x.dtype, "int32")

        self.assertIsInstance(func_right_x, KerasTensor)
        self.assertEqual(func_right_x.shape, ())
        self.assertEqual(func_right_x.dtype, "int32")

        self.assertNotIsInstance(eager_x, KerasTensor)
        self.assertTrue(ops.is_tensor(eager_x))

        left_model = Functional(inputs=left_input, outputs=func_left_x)
        left_model_x = left_model(ops.convert_to_tensor(1))
        self.assertNotIsInstance(left_model_x, KerasTensor)
        self.assertTrue(ops.is_tensor(left_model_x))
        self.assertAllClose(left_model_x, eager_x)

        right_model = Functional(inputs=right_input, outputs=func_right_x)
        right_model_x = right_model(ops.convert_to_tensor(3))
        self.assertNotIsInstance(right_model_x, KerasTensor)
        self.assertTrue(ops.is_tensor(right_model_x))
        self.assertAllClose(right_model_x, eager_x)

        inputs = Input(batch_shape=(), dtype="int32")
        outputs = ops.add(scope_x, inputs)
        model = Functional(inputs=inputs, outputs=outputs)

        self.verify_model(model, ops.zeros((), dtype="int32"), eager_x)
