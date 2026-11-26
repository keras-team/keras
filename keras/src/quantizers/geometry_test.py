import numpy as np
from absl.testing import parameterized

from keras.src import layers
from keras.src import testing
from keras.src.layers.core.einsum_dense import EinsumAxes
from keras.src.quantizers.geometry import ContractionView
from keras.src.quantizers.geometry import KernelAxes


def _expert_slices(x):
    # `btei,eid->bted`: expert `e` sees `x[:, :, e, :]`.
    return np.stack(
        [x[:, :, e, :].reshape(-1, x.shape[-1]) for e in range(x.shape[2])]
    )


def _head_slices(x):
    # `btnh,nh->btn`: head `n` sees `x[:, :, n, :]`.
    return x.transpose(2, 0, 1, 3).reshape(x.shape[2], -1, x.shape[-1])


class ContractionViewTest(testing.TestCase):
    """The contraction view follows the equation, not the kernel shape."""

    @parameterized.named_parameters(
        (
            "qkv",
            "btd,dnh->btnh",
            (None, 2, 3),
            (None, 5, 4),
            (1, 4, 6),
            (0, 1, 2),
            lambda x: x.reshape(-1, 4),
        ),
        (
            "attention_output",
            "btnh,nhd->btd",
            (None, 4),
            (None, 5, 2, 3),
            (1, 6, 4),
            (0, 1, 2),
            lambda x: x.reshape(-1, 6),
        ),
        (
            "gemma_q",
            "btd,ndh->btnh",
            (None, 2, 3),
            (None, 5, 4),
            (1, 4, 6),
            (1, 0, 2),
            lambda x: x.reshape(-1, 4),
        ),
        (
            "expert_down",
            "btei,eid->bted",
            (None, 3, 4),
            (None, 5, 3, 6),
            (3, 6, 4),
            (0, 1, 2),
            _expert_slices,
        ),
        (
            # The kernel orders its batch axes `[h, g]` and the inputs
            # `[g, h]`: problem `(h, g)` sees `x[:, g, h, :]`.
            "two_batch_axes",
            "bghc,hgcd->bghd",
            (2, 3, 5),
            (None, 2, 3, 4),
            (6, 4, 5),
            (0, 1, 2, 3),
            lambda x: x.transpose(2, 1, 0, 3).reshape(6, -1, 4),
        ),
        (
            # The inputs carry the contracted axes in the other order: the
            # inputs view follows the kernel's `[n, h]`.
            "input_order_differs",
            "bhn,nhd->bd",
            (4,),
            (None, 2, 3),
            (1, 6, 4),
            (0, 1, 2),
            lambda x: x.transpose(0, 2, 1).reshape(-1, 6),
        ),
        (
            "ellipsis",
            "...d,dnh->...nh",
            (2, 3),
            (None, 5, 4),
            (1, 4, 6),
            (0, 1, 2),
            lambda x: x.reshape(-1, 4),
        ),
        (
            "four_d",
            "abc,cdef->abdef",
            (3, 2, 3, 2),
            (None, 3, 4),
            (1, 4, 12),
            (0, 1, 2, 3),
            lambda x: x.reshape(-1, 4),
        ),
        (
            # A value head: the kernel has no free axis, so one column.
            "no_free_axis",
            "btd,d->bt",
            (None,),
            (None, 5, 4),
            (1, 4, 1),
            (0,),
            lambda x: x.reshape(-1, 4),
        ),
        (
            # A per-head gate: `n` is a batch axis and `h` contracted.
            "batch_axis_without_free_axis",
            "btnh,nh->btn",
            (None, 3),
            (None, 5, 3, 4),
            (3, 4, 1),
            (0, 1),
            _head_slices,
        ),
    )
    def test_einsum_view(
        self,
        equation,
        output_shape,
        input_shape,
        expected_dims,
        expected_permutation,
        expected_inputs_view,
    ):
        layer = layers.EinsumDense(equation, output_shape=output_shape)
        layer.build(input_shape)
        view = layer._quantization_geometry().contraction_view()
        batch, rows, columns = expected_dims
        self.assertEqual((view.batch, view.rows, view.columns), expected_dims)
        self.assertEqual(view.kernel_permutation, expected_permutation)

        # The kernel view flattens the permuted axes and restores exactly.
        kernel_shape = tuple(layer.kernel.shape)
        kernel = np.arange(np.prod(kernel_shape), dtype="float32").reshape(
            kernel_shape
        )
        kernel_view = view.kernel_to_view(kernel)
        self.assertEqual(tuple(kernel_view.shape), (batch, rows, columns))
        self.assertAllClose(
            kernel_view,
            np.transpose(kernel, expected_permutation).reshape(
                batch, rows, columns
            ),
        )

        # The inputs view lines its rows up with the kernel view's rows.
        x = (
            np.random.default_rng(0)
            .standard_normal((2,) + tuple(input_shape[1:]))
            .astype("float32")
        )
        self.assertEqual(view.input_features(x), rows)
        self.assertAllClose(view.inputs_to_view(x), expected_inputs_view(x))

    @parameterized.named_parameters(
        ("missing", (0,), (1,)),
        ("duplicated", (0, 1), (1, 2)),
    )
    def test_axes_must_partition_the_kernel(self, contracted, free):
        kernel_axes = KernelAxes(contracted=contracted, free=free)
        with self.assertRaisesRegex(
            ValueError, r"partition the kernel axes.*rank-3.*kernel_axes="
        ):
            ContractionView(
                (2, 3, 4),
                kernel_axes,
                input_batch_axes=(),
                input_contracted_axes=(-1,),
            )


class KernelAxesTest(testing.TestCase):
    def test_axes_are_stored_as_tuples(self):
        self.assertEqual(
            KernelAxes(contracted=[0, 1], free=[2]),
            KernelAxes(contracted=(0, 1), free=(2,)),
        )


class EinsumAxesTest(testing.TestCase):
    @parameterized.named_parameters(
        ("left_ellipsis", "...d,dh->...h", 4, "ABCd", "ABCh", (0,), (1,), ()),
        (
            "right_ellipsis",
            "ab...,bc->ac...",
            4,
            "abAB",
            "acAB",
            (0,),
            (1,),
            (),
        ),
    )
    def test_labels_and_roles(
        self, equation, rank, inputs, output, contracted, free, batch
    ):
        axes = EinsumAxes.from_equation(equation, rank)
        self.assertEqual((axes.inputs, axes.output), (inputs, output))
        self.assertEqual(axes.kernel_axes, KernelAxes(contracted, free, batch))
        # The input axes that carry the kernel's contracted labels are the
        # ones the equation reduces.
        self.assertEqual(axes.input_axes(contracted), axes.input_reduced_axes)
        self.assertEqual(
            axes.gradient_equation, f"{output},{axes.kernel}->{inputs}"
        )
