import pytest

from keras.src import backend
from keras.src import testing


def example_fn(x):
    x = (x + 2) * backend.numpy.ones_like(x)
    x = backend.numpy.stack([x, x], axis=-1)
    return x


class ComputeOutputSpecTest(testing.TestCase):
    def test_basics(self):
        out = backend.compute_output_spec(
            example_fn, backend.KerasTensor((2, 3))
        )
        self.assertIsInstance(out, backend.KerasTensor)
        self.assertEqual(out.shape, (2, 3, 2))

        out = backend.compute_output_spec(
            example_fn, backend.KerasTensor((None, 3))
        )
        self.assertIsInstance(out, backend.KerasTensor)
        self.assertEqual(out.shape, (None, 3, 2))

        out = backend.compute_output_spec(
            example_fn, backend.KerasTensor((2, None))
        )
        self.assertIsInstance(out, backend.KerasTensor)
        self.assertEqual(out.shape, (2, None, 2))

    @pytest.mark.skipif(
        backend.backend() != "torch", reason="Only applicable for torch"
    )
    def test_torch_meta_device_incompatible_ops(self):
        class Container:
            def __init__(self):
                self.canary = False

            def example_meta_fn(self, x):
                y = backend.numpy.ones(x.shape)
                if str(y.device) == "meta":
                    self.canary = True
                    raise ValueError("Erroring out on meta device")
                x = (x + 2) * y
                x = backend.numpy.stack([x, x], axis=-1)
                return x

        instance = Container()
        out = backend.compute_output_spec(
            instance.example_meta_fn, backend.KerasTensor((2, 3))
        )
        self.assertIsInstance(out, backend.KerasTensor)
        self.assertTrue(instance.canary)
        self.assertEqual(out.shape, (2, 3, 2))

        instance = Container()
        out = backend.compute_output_spec(
            instance.example_meta_fn, backend.KerasTensor((2, None))
        )
        self.assertIsInstance(out, backend.KerasTensor)
        self.assertTrue(instance.canary)
        self.assertEqual(out.shape, (2, None, 2))
