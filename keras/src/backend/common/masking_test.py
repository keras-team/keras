from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.common.masking import get_keras_mask
from keras.src.backend.common.masking import set_keras_mask


class MaskingTest(testing.TestCase):
    def test_mask_on_eager_tensor(self):
        x = ops.zeros((2, 3))
        self.assertIsNone(get_keras_mask(x))

        set_keras_mask(x, None)
        self.assertIsNone(get_keras_mask(x))

        mask = ops.ones((2, 3))
        set_keras_mask(x, mask)
        self.assertIs(get_keras_mask(x), mask)

        set_keras_mask(x, None)
        self.assertIsNone(get_keras_mask(x))

        set_keras_mask(x, None)
        self.assertIsNone(get_keras_mask(x))

    def test_mask_on_tracer_tensor(self):
        def fn(x):
            self.assertIsNone(get_keras_mask(x))

            set_keras_mask(x, None)
            self.assertIsNone(get_keras_mask(x))

            mask = ops.ones((2, 3))
            set_keras_mask(x, mask)
            self.assertIs(get_keras_mask(x), mask)

            set_keras_mask(x, None)
            self.assertIsNone(get_keras_mask(x))

            set_keras_mask(x, None)  # key is now deleted, should be a no-op
            self.assertIsNone(get_keras_mask(x))

        backend.compute_output_spec(fn, backend.KerasTensor((2, 3)))
