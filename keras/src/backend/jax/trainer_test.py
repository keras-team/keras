import queue
import warnings
from unittest import mock

import jax
import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend import distribution_lib as backend_dlib
from keras.src.backend.jax import trainer as jax_trainer
from keras.src.distribution import distribution_lib


@pytest.mark.skipif(backend.backend() != "jax", reason="JAX only")
class JAXEpochIteratorThreadedTest(testing.TestCase):
    def test_device_put_tree(self):
        batch = (np.ones((2, 3), dtype="float32"), None)

        result = jax_trainer.device_put_tree(batch)

        self.assertEqual(result[0].shape, (2, 3))
        self.assertIsNone(result[1])

    def test_shape_signature(self):
        batch = {
            "x": np.ones((2, 3), dtype="float32"),
            "none": None,
        }

        signature = jax_trainer.shape_signature(batch)
        leaf_signatures = [leaf_signature for _, leaf_signature in signature]

        self.assertIn(((2, 3), np.dtype("float32")), leaf_signatures)
        self.assertIn(None, leaf_signatures)

    def test_threaded_prefetch_iterator_yields_transformed_items(self):
        iterator = jax_trainer.ThreadedPrefetchIterator(
            [1, 2, 3],
            transform=lambda x: x + 1,
            maxsize=1,
        )

        self.assertIs(iter(iterator), iterator)
        self.assertEqual(next(iterator), 2)
        self.assertEqual(next(iterator), 3)
        self.assertEqual(next(iterator), 4)
        with self.assertRaises(StopIteration):
            next(iterator)
        self.assertTrue(iterator.closed)

    def test_threaded_prefetch_iterator_producer_stops_when_queue_full(self):
        iterator = jax_trainer.ThreadedPrefetchIterator(
            [1],
            transform=lambda x: x,
        )

        def put_and_stop(*args, **kwargs):
            iterator.stop.set()
            raise queue.Full

        with mock.patch.object(
            iterator.queue,
            "put",
            side_effect=put_and_stop,
        ) as queue_put:
            iterator._producer()

        self.assertEqual(queue_put.call_count, 1)
        self.assertEqual(iterator.errors, [])

    def test_threaded_prefetch_iterator_raises_transform_errors(self):
        def transform(x):
            if x == 2:
                raise ValueError("failed")
            return x

        iterator = jax_trainer.ThreadedPrefetchIterator(
            [1, 2],
            transform=transform,
        )

        self.assertEqual(next(iterator), 1)
        with self.assertRaisesRegex(ValueError, "failed"):
            next(iterator)
        self.assertTrue(iterator.closed)

    def test_threaded_prefetch_iterator_producer_captures_base_exceptions(self):
        error = BaseException("failed")

        def transform(x):
            raise error

        iterator = jax_trainer.ThreadedPrefetchIterator(
            [1],
            transform=transform,
        )
        iterator._producer()

        self.assertEqual(iterator.errors, [error])
        self.assertIs(iterator.queue.get_nowait(), iterator.end)

    def test_stopped_threaded_prefetch_iterator_producer_ignores_errors(self):
        def transform(x):
            iterator.stop.set()
            raise BaseException("failed")

        iterator = jax_trainer.ThreadedPrefetchIterator(
            [1],
            transform=transform,
        )
        iterator._producer()

        self.assertEqual(iterator.errors, [])
        with self.assertRaises(queue.Empty):
            iterator.queue.get_nowait()

    def test_threaded_prefetch_iterator_close(self):
        def infinite_iterator():
            while True:
                yield 1

        iterator = jax_trainer.ThreadedPrefetchIterator(
            infinite_iterator(),
            transform=lambda x: x,
            maxsize=1,
        )

        self.assertEqual(next(iterator), 1)
        iterator.close()
        iterator.close()
        self.assertTrue(iterator.closed)
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_jax_epoch_iterator_threaded_reset_closes_current_iterator(self):
        iterator = jax_trainer.JAXEpochIteratorThreaded(
            x=np.ones((2, 1)),
            batch_size=1,
        )
        current_iterator = mock.Mock()
        iterator._current_iterator = current_iterator
        iterator._epoch_iterator = object()
        iterator._steps_seen = 1

        with mock.patch.object(
            iterator.data_adapter, "on_epoch_end"
        ) as on_epoch_end:
            iterator.reset()

        current_iterator.close.assert_called_once()
        on_epoch_end.assert_called_once()
        self.assertIsNone(iterator._current_iterator)
        self.assertIsNone(iterator._epoch_iterator)
        self.assertEqual(iterator._steps_seen, 0)

    def test_jax_epoch_iterator_threaded_close_clears_current_iterator(self):
        iterator = jax_trainer.JAXEpochIteratorThreaded(
            x=np.ones((2, 1)),
            batch_size=1,
        )
        current_iterator = mock.Mock()
        iterator._current_iterator = current_iterator

        iterator.close()

        current_iterator.close.assert_called_once()
        self.assertIsNone(iterator._current_iterator)

        iterator._current_iterator = object()

        iterator.close()

        self.assertIsNone(iterator._current_iterator)

    def test_jax_epoch_iterator_threaded_get_iterator(self):
        iterator = jax_trainer.JAXEpochIteratorThreaded(
            x=np.ones((2, 1)),
            batch_size=1,
            prefetch=3,
        )
        jax_iterator = object()
        prepare_batch = object()
        threaded_iterator = object()

        with (
            mock.patch.object(
                jax_trainer.distribution_lib,
                "distribution",
                return_value=None,
            ) as distribution,
            mock.patch.object(
                iterator,
                "_make_prepare_batch",
                return_value=prepare_batch,
            ) as make_prepare_batch,
            mock.patch.object(
                iterator.data_adapter,
                "get_jax_iterator",
                return_value=jax_iterator,
            ) as get_jax_iterator,
            mock.patch.object(
                jax_trainer,
                "ThreadedPrefetchIterator",
                return_value=threaded_iterator,
            ) as threaded_prefetch_iterator,
        ):
            result = iterator._get_iterator()

        distribution.assert_called_once_with()
        make_prepare_batch.assert_called_once_with(None)
        get_jax_iterator.assert_called_once_with()
        threaded_prefetch_iterator.assert_called_once_with(
            jax_iterator,
            transform=prepare_batch,
            maxsize=3,
        )
        self.assertIs(result, threaded_iterator)

    def test_build_jax_epoch_iterator_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            iterator = jax_trainer.build_jax_epoch_iterator(
                x=np.ones((2, 1)),
                batch_size=1,
            )

        self.assertIsInstance(iterator, jax_trainer.JAXEpochIterator)
        self.assertNotIsInstance(iterator, jax_trainer.JAXEpochIteratorThreaded)

    def test_build_jax_epoch_iterator_threaded(self):
        with mock.patch.dict(
            "os.environ", {"KERAS_JAX_EPOCH_ITERATOR": "threaded"}
        ):
            iterator = jax_trainer.build_jax_epoch_iterator(
                x=np.ones((2, 1)),
                batch_size=1,
            )

        self.assertIsInstance(iterator, jax_trainer.JAXEpochIteratorThreaded)
        iterator.close()

    def test_build_jax_epoch_iterator_invalid(self):
        with mock.patch.dict(
            "os.environ", {"KERAS_JAX_EPOCH_ITERATOR": "invalid"}
        ):
            with self.assertRaisesRegex(ValueError, "Invalid value"):
                jax_trainer.build_jax_epoch_iterator(
                    x=np.ones((2, 1)),
                    batch_size=1,
                )

    def test_batch_preparation_without_distribution(self):
        iterator = jax_trainer.JAXEpochIteratorThreaded(
            x=np.ones((2, 1)),
            batch_size=1,
        )

        prepare_batch = iterator._make_prepare_batch(None)

        self.assertIs(prepare_batch, jax_trainer.device_put_tree)

    def test_distributed_batch_preparation_caches_layouts(self):
        class DataLayout:
            def __init__(self, shape):
                self.backend_layout = ("layout", tuple(shape))

        class Distribution:
            def __init__(self):
                self.calls = 0

            def get_data_layout(self, shape):
                self.calls += 1
                return DataLayout(shape)

        distribution = Distribution()
        iterator = jax_trainer.JAXEpochIteratorThreaded(
            x=np.ones((2, 1)),
            batch_size=1,
        )
        prepare_batch = iterator._make_prepare_batch(distribution)

        with mock.patch.object(
            jax_trainer,
            "_distribute_data",
            side_effect=lambda batch, layouts: (batch, layouts),
        ):
            batch = (np.ones((2, 3), dtype="float32"), None)
            _, layouts = prepare_batch(batch)
            self.assertEqual(layouts[0], ("layout", (2, 3)))
            self.assertIsNone(layouts[1])
            self.assertEqual(distribution.calls, 1)

            self.assertIsNone(prepare_batch(None))
            self.assertEqual(distribution.calls, 1)

            prepare_batch((np.ones((2, 3), dtype="float32"), None))
            self.assertEqual(distribution.calls, 1)

            prepare_batch((np.ones((4, 3), dtype="float32"), None))
            self.assertEqual(distribution.calls, 2)

            prepare_batch((np.ones((2, 3), dtype="float32"), None))
            self.assertEqual(distribution.calls, 2)


@pytest.mark.skipif(backend.backend() != "jax", reason="JAX only")
@pytest.mark.multi_device
class JAXTrainerTest(testing.TestCase, parameterized.TestCase):
    def _make_distribution(self, dist_type):
        if dist_type == "data_parallel":
            return distribution_lib.DataParallel()
        devices = backend_dlib.list_devices()
        n = len(devices)
        mesh = distribution_lib.DeviceMesh((n,), ["model"], devices)
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        layout_map[".*dense.*bias"] = distribution_lib.TensorLayout(["model"])
        return distribution_lib.ModelParallel(layout_map=layout_map)

    # ----------------------------------------------------------------
    # Mixed-sharding warning tests
    # ----------------------------------------------------------------
    @parameterized.named_parameters(
        {"testcase_name": "DataParallel", "dist_type": "data_parallel"},
        {"testcase_name": "ModelParallel", "dist_type": "model_parallel"},
    )
    def test_warns_when_model_built_outside_scope(self, dist_type):
        """Model built outside distribution -> mixed warning on compile."""
        n = len(backend_dlib.list_devices())
        units = n * max(1, 4 // n)
        dist = self._make_distribution(dist_type)

        # Model created outside any distribution scope — weights are local.
        model = models.Sequential([layers.Dense(units, input_shape=(16,))])

        for w in model.weights:
            self.assertIsInstance(
                w.value.sharding, jax.sharding.SingleDeviceSharding
            )

        inputs = np.random.normal(size=(8, 16)).astype("float32")
        labels = np.random.normal(size=(8, units)).astype("float32")

        with dist.scope():
            model.compile(loss="mse", optimizer="adam")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model._symbolic_build(data_batch=(inputs[:2], labels[:2]))
                model._get_state_sharding_spec()

            mixed = [w for w in caught if "mix of local" in str(w.message)]
            self.assertGreater(
                len(mixed),
                0,
                "Expected a mixed-sharding warning but none was raised",
            )
            msg = str(mixed[0].message)
            self.assertIn("SingleDeviceSharding", msg)
            self.assertIn("set_distribution", msg)

    @parameterized.named_parameters(
        {"testcase_name": "DataParallel", "dist_type": "data_parallel"},
        {"testcase_name": "ModelParallel", "dist_type": "model_parallel"},
    )
    def test_no_warning_when_model_built_inside_scope(self, dist_type):
        """Model built inside distribution scope -> no warning."""
        n = len(backend_dlib.list_devices())
        units = n * max(1, 4 // n)
        dist = self._make_distribution(dist_type)

        # Model created inside scope — weights get proper sharding.
        with dist.scope():
            model = models.Sequential([layers.Dense(units, input_shape=(16,))])

        inputs = np.random.normal(size=(8, 16)).astype("float32")
        labels = np.random.normal(size=(8, units)).astype("float32")

        with dist.scope():
            model.compile(loss="mse", optimizer="adam")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model._symbolic_build(data_batch=(inputs[:2], labels[:2]))
                model._get_state_sharding_spec()

            mixed = [w for w in caught if "mix of local" in str(w.message)]
            self.assertEqual(
                len(mixed),
                0,
                "Unexpected mixed-sharding warning when model is "
                "built inside scope",
            )

    @parameterized.named_parameters(
        {"testcase_name": "DataParallel", "dist_type": "data_parallel"},
        {"testcase_name": "ModelParallel", "dist_type": "model_parallel"},
    )
    def test_train_on_batch(self, dist_type):
        n = len(backend_dlib.list_devices())
        units = n * max(1, 4 // n)
        dist = self._make_distribution(dist_type)

        with dist.scope():
            model = models.Sequential([layers.Dense(units, input_shape=(16,))])
            model.compile(loss="mse", optimizer="adam")

            inputs = np.random.normal(size=(8, 16)).astype("float32")
            labels = np.random.normal(size=(8, units)).astype("float32")
            sw = np.random.uniform(size=(8,)).astype("float32")

            # With sample weight.
            model.train_on_batch(x=inputs, y=labels, sample_weight=sw)
            model.test_on_batch(x=inputs, y=labels, sample_weight=sw)

            # Without sample weight.
            model.train_on_batch(x=inputs, y=labels)
            model.test_on_batch(x=inputs, y=labels)
            model.predict_on_batch(x=inputs)
            model.fit(x=inputs, y=labels, epochs=1, verbose=0)
            model.evaluate(x=inputs, y=labels, verbose=0)

    @parameterized.named_parameters(
        {"testcase_name": "DataParallel", "dist_type": "data_parallel"},
        {"testcase_name": "ModelParallel", "dist_type": "model_parallel"},
    )
    def test_jax_epoch_iterator_with_none_elements(self, dist_type):
        def generator():
            yield (np.ones((16, 32)), None)

        with self._make_distribution(dist_type).scope():
            iterator = jax_trainer.JAXEpochIterator(
                x=generator(), steps_per_epoch=1
            )

            epoch_iter = iterator._get_iterator()
            batch = next(epoch_iter)

        self.assertIsNone(batch[1])
        self.assertIsNotNone(batch[0])
