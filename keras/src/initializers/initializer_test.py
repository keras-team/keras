import jax
import pytest

from keras.src import backend
from keras.src import initializers
from keras.src import testing
from keras.src.distribution import distribution_lib


class InitializerWithLayout(initializers.Initializer):
    def __call__(self, shape, dtype=None, layout=None):
        if isinstance(layout, distribution_lib.TensorLayout):
            layout = layout.backend_layout
        return jax.numpy.ones(shape, dtype=dtype, out_sharding=layout)


@pytest.mark.skipif(backend.backend() != "jax", reason="JAX only")
@pytest.mark.multi_device
class InitializersLayoutTest(testing.TestCase):
    def test_initializer_with_data_parallel_layout(self):
        distribution = distribution_lib.DataParallel()

        shape = (2, 3)
        expected_sharding = distribution.get_variable_layout(
            backend.Variable("zeros", shape, "float32")
        ).backend_layout

        with distribution.scope():
            var = backend.Variable(InitializerWithLayout(), shape, "float32")

        self.assertAllClose(var.value, 1)
        self.assertTrue(
            var.value.sharding.is_equivalent_to(expected_sharding, len(shape)),
            msg=f"Actual: {var.value.sharding}, Expected: {expected_sharding}",
        )

    def test_initializer_with_model_parallel_layout(self):
        device_count = distribution_lib.get_device_count()
        self.assertGreaterEqual(
            device_count, 4, "Number of devices must be at least 4"
        )
        self.assertEqual(device_count % 2, 0, "Number of devices must be even")
        mesh_shape = (device_count // 2, 2)
        device_mesh = distribution_lib.DeviceMesh(
            mesh_shape, axis_names=("batch", "model")
        )

        layout_map = distribution_lib.LayoutMap(device_mesh)
        layout_map["kernel"] = distribution_lib.TensorLayout([None, "model"])
        layout_map["bias"] = distribution_lib.TensorLayout(["model"])
        distribution = distribution_lib.ModelParallel(layout_map=layout_map)

        shape = (2, 4)
        expected_kernel_sharding = distribution.get_variable_layout(
            backend.Variable("zeros", shape, "float32", name="kernel")
        ).backend_layout
        expected_bias_sharding = distribution.get_variable_layout(
            backend.Variable("zeros", shape, "float32", name="bias")
        ).backend_layout

        with distribution.scope():
            kernel = backend.Variable(
                InitializerWithLayout(), shape, "float32", name="kernel"
            )
            bias = backend.Variable(
                InitializerWithLayout(), shape, "float32", name="bias"
            )

        self.assertAllClose(kernel.value, 1)
        self.assertTrue(
            kernel.value.sharding.is_equivalent_to(
                expected_kernel_sharding, len(shape)
            ),
            msg=(
                f"Actual: {kernel.value.sharding}, "
                f"Expected: {expected_kernel_sharding}"
            ),
        )

        self.assertAllClose(bias.value, 1)
        self.assertTrue(
            bias.value.sharding.is_equivalent_to(
                expected_bias_sharding, len(shape)
            ),
            msg=(
                f"Actual: {bias.value.sharding}, "
                f"Expected: {expected_bias_sharding}"
            ),
        )
