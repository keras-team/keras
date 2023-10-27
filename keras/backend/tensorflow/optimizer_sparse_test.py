from unittest import mock

import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras import backend
from keras import optimizers
from keras import testing

TEST_CASES = [
    {
        "testcase_name": "adadelta",
        "optimizer_class": optimizers.Adadelta,
        "expect_model_sparse_variable_updates": True,
    },
    {
        "testcase_name": "adafactor",
        "optimizer_class": optimizers.Adafactor,
        "init_kwargs": {"clip_threshold": 0.5},
        "expect_model_sparse_variable_updates": True,
    },
    {
        "testcase_name": "adagrad",
        "optimizer_class": optimizers.Adagrad,
        "expect_model_sparse_variable_updates": True,
        "expect_optimizer_sparse_variable_updates": True,
    },
    {
        "testcase_name": "adam",
        "optimizer_class": optimizers.Adam,
    },
    {
        "testcase_name": "adam_amsgrad",
        "optimizer_class": optimizers.Adam,
        "init_kwargs": {"amsgrad": True},
    },
    {
        "testcase_name": "adamax",
        "optimizer_class": optimizers.Adamax,
    },
    {
        "testcase_name": "adamw",
        "optimizer_class": optimizers.AdamW,
    },
    {
        "testcase_name": "adamw_amsgrad",
        "optimizer_class": optimizers.AdamW,
        "init_kwargs": {"amsgrad": True},
    },
    {
        "testcase_name": "ftrl",
        "optimizer_class": optimizers.Ftrl,
    },
    {
        "testcase_name": "lion",
        "optimizer_class": optimizers.Lion,
    },
    {
        "testcase_name": "loss_scale_optimizer_sgd",
        "optimizer_class": lambda: optimizers.LossScaleOptimizer(
            optimizers.SGD(learning_rate=0.5)
        ),
        "expect_model_sparse_variable_updates": True,
    },
    {
        "testcase_name": "nadam",
        "optimizer_class": optimizers.Nadam,
    },
    {
        "testcase_name": "rmsprop",
        "optimizer_class": optimizers.RMSprop,
        "expect_model_sparse_variable_updates": True,
    },
    {
        "testcase_name": "rmsprop_momentum",
        "optimizer_class": optimizers.RMSprop,
        "init_kwargs": {"momentum": 0.05},
    },
    {
        "testcase_name": "rmsprop_momentum_centered",
        "optimizer_class": optimizers.RMSprop,
        "init_kwargs": {"momentum": 0.05, "centered": True},
    },
    {
        "testcase_name": "sgd",
        "optimizer_class": optimizers.SGD,
        "expect_model_sparse_variable_updates": True,
    },
    {
        "testcase_name": "sgd_momentum",
        "optimizer_class": optimizers.SGD,
        "init_kwargs": {"momentum": 0.05},
    },
    {
        "testcase_name": "sgd_momentum_nesterov",
        "optimizer_class": optimizers.SGD,
        "init_kwargs": {"momentum": 0.05, "nesterov": True},
    },
]


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="The TensorFlow sparse test can only run with TensorFlow backend.",
)
class OptimizerSparseTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(TEST_CASES)
    def test_sparse_gradients(
        self,
        optimizer_class,
        init_kwargs={},
        expect_model_sparse_variable_updates=False,
        expect_optimizer_sparse_variable_updates=False,
    ):
        # This test verifies that:
        # - Optimizers use Keras ops everywhere instead of native operators
        #   (e.g. `ops.add()` instead of `+`) where sparse gradients are handled
        # - The used ops handle sparse gradients (`tf.IndexedSlices`)
        # - Optimizers use `self.assign/assign_add/assign_sub` instead of
        #   calling the method on the variable directly. Otherwise, the sparse
        #   updates are densified before being applied.
        # - For some optimizers, a sparse gradient actually results in a sparse
        #   variable update as per `expect_model_sparse_variable_updates` and
        #   `expect_optimizer_sparse_variable_updates`

        model_variable = backend.Variable(initializer=tf.ones, shape=(5, 10))
        optimizer = optimizer_class(**init_kwargs)

        # Mocking "tensorflow.Variable" won't work as it gets substituted with
        # the resource variable class.
        tf_variable_class = model_variable._value.__class__

        optimizer_to_patch = (
            optimizer.inner_optimizer
            if isinstance(optimizer, optimizers.LossScaleOptimizer)
            else optimizer
        )

        model_sparse_variable_updates = False
        optimizer_sparse_variable_updates = False

        def mock_optimizer_assign(variable, value):
            nonlocal model_sparse_variable_updates
            nonlocal optimizer_sparse_variable_updates
            if isinstance(variable, backend.Variable):
                variable = variable._value
            if isinstance(value, tf.IndexedSlices):
                if variable is model_variable._value:
                    model_sparse_variable_updates = True
                elif any(variable is v._value for v in optimizer.variables):
                    optimizer_sparse_variable_updates = True

        def mock_variable_assign(variable, value):
            # Make an exception for scalar variables
            if len(variable.shape):
                pytest.fail(
                    "Optimizer is calling `assign`, `assign_add` or "
                    "`assign_sub` directly on a variable. Use "
                    "`self.assign/assign_add/assign_sub(variable, value)` "
                    "instead to support sparse updates."
                )

        # patch "_apply_weight_decay" to exclude this special case.
        # patch the optimizer "assign" methods to detect sparse udpates.
        # patch the tf.Variable "assign" methods to detect direct assign calls.
        with mock.patch.object(
            optimizer_to_patch, "_apply_weight_decay", autospec=True
        ), mock.patch.object(
            optimizer_to_patch, "assign", autospec=True
        ) as optimizer_assign, mock.patch.object(
            optimizer_to_patch, "assign_add", autospec=True
        ) as optimizer_assign_add, mock.patch.object(
            optimizer_to_patch, "assign_sub", autospec=True
        ) as optimizer_assign_sub, mock.patch.object(
            tf_variable_class, "assign", autospec=True
        ) as variable_assign, mock.patch.object(
            tf_variable_class, "assign_add", autospec=True
        ) as variable_assign_add, mock.patch.object(
            tf_variable_class, "assign_sub", autospec=True
        ) as variable_assign_sub:
            optimizer_assign.side_effect = mock_optimizer_assign
            optimizer_assign_add.side_effect = mock_optimizer_assign
            optimizer_assign_sub.side_effect = mock_optimizer_assign
            variable_assign.side_effect = mock_variable_assign
            variable_assign_add.side_effect = mock_variable_assign
            variable_assign_sub.side_effect = mock_variable_assign

            grad = tf.IndexedSlices(
                values=tf.ones((3, 10)), indices=(0, 2, 4), dense_shape=(5, 10)
            )
            optimizer.apply_gradients(zip([grad], [model_variable]))

        self.assertEqual(
            model_sparse_variable_updates, expect_model_sparse_variable_updates
        )
        self.assertEqual(
            optimizer_sparse_variable_updates,
            expect_optimizer_sparse_variable_updates,
        )

    @parameterized.named_parameters(TEST_CASES)
    def test_sparse_correctness(
        self, optimizer_class, init_kwargs={}, **kwargs
    ):
        # This test verifies that applying a sparse gradient gives the same
        # numerical results as the same dense gradient.

        optimizer_sparse = optimizer_class(**init_kwargs)
        optimizer_dense = optimizer_class(**init_kwargs)
        var_sparse = backend.Variable(initializer=tf.ones, shape=(5, 3, 2))
        var_dense = backend.Variable(initializer=tf.ones, shape=(5, 3, 2))

        for i in range(5):
            grads_sparse = tf.IndexedSlices(
                values=tf.ones((3, 3, 2)) * (10.0 - i),
                indices=(0, 2, 4),
                dense_shape=(5, 3, 2),
            )
            grads_dense = tf.convert_to_tensor(grads_sparse)
            optimizer_sparse.apply_gradients(zip([grads_sparse], [var_sparse]))
            optimizer_dense.apply_gradients(zip([grads_dense], [var_dense]))
            self.assertAllClose(var_sparse, var_dense)
