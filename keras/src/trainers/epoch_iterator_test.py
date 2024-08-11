import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.trainers import data_adapters
from keras.src.trainers import epoch_iterator


class TestEpochIterator(testing.TestCase, parameterized.TestCase):
    def test_basic_flow(self):
        x = np.random.random((100, 16))
        y = np.random.random((100, 4))
        sample_weight = np.random.random((100,))
        batch_size = 16
        shuffle = True
        iterator = epoch_iterator.EpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        steps_seen = []
        for step, batch in iterator.enumerate_epoch():
            batch = batch[0]
            steps_seen.append(step)
            self.assertEqual(len(batch), 3)
            self.assertIsInstance(batch[0], np.ndarray)
        self.assertEqual(steps_seen, [0, 1, 2, 3, 4, 5, 6])

    def test_insufficient_data(self):
        batch_size = 8
        steps_per_epoch = 6
        dataset_size = batch_size * (steps_per_epoch - 2)
        x = np.arange(dataset_size).reshape((dataset_size, 1))
        y = x * 2
        iterator = epoch_iterator.EpochIterator(
            x=x,
            y=y,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
        )
        steps_seen = []
        for step, _ in iterator.enumerate_epoch():
            steps_seen.append(step)
        self.assertLen(steps_seen, steps_per_epoch - 2)

        self.assertIsInstance(iterator, epoch_iterator.EpochIterator)
        self.assertTrue(iterator._insufficient_data)

    def test_unsupported_y_arg_tfdata(self):
        with self.assertRaisesRegex(ValueError, "`y` should not be passed"):
            x = tf.data.Dataset.from_tensor_slices(np.random.random((100, 16)))
            y = np.random.random((100, 4))
            _ = epoch_iterator.EpochIterator(x=x, y=y)

    def test_unsupported_sample_weights_arg_tfdata(self):
        with self.assertRaisesRegex(
            ValueError, "`sample_weights` should not be passed"
        ):
            x = tf.data.Dataset.from_tensor_slices(np.random.random((100, 16)))
            sample_weights = np.random.random((100,))
            _ = epoch_iterator.EpochIterator(x=x, sample_weight=sample_weights)

    @pytest.mark.skipif(
        backend.backend() != "torch", reason="Need to import torch"
    )
    def test_torch_dataloader(self):
        import torch

        class ExampleTorchDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        torch_dataset = ExampleTorchDataset(
            np.random.random((64, 2)), np.random.random((64, 1))
        )
        torch_dataloader = torch.utils.data.DataLoader(
            torch_dataset, batch_size=8, shuffle=True
        )
        iterator = epoch_iterator.EpochIterator(torch_dataloader)
        for _, batch in iterator.enumerate_epoch():
            batch = batch[0]
            self.assertEqual(batch[0].shape, (8, 2))
            self.assertEqual(batch[1].shape, (8, 1))

    @pytest.mark.skipif(
        backend.backend() != "torch", reason="Need to import torch"
    )
    def test_unsupported_y_arg_torch_dataloader(self):
        import torch

        class ExampleTorchDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        torch_dataset = ExampleTorchDataset(
            np.random.random((100, 16)), np.random.random((100, 4))
        )
        x = torch.utils.data.DataLoader(
            torch_dataset, batch_size=8, shuffle=True
        )
        y = np.random.random((100, 4))

        with self.assertRaisesRegex(
            ValueError,
            "When providing `x` as a torch DataLoader, `y` should not",
        ):
            _ = epoch_iterator.EpochIterator(x=x, y=y)

    @pytest.mark.skipif(
        backend.backend() != "torch", reason="Need to import torch"
    )
    def test_unsupported_sample_weights_arg_torch_dataloader(self):
        import torch

        class ExampleTorchDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        torch_dataset = ExampleTorchDataset(
            np.random.random((100, 16)), np.random.random((100, 4))
        )
        x = torch.utils.data.DataLoader(
            torch_dataset, batch_size=8, shuffle=True
        )
        sample_weights = np.random.random((100,))

        with self.assertRaisesRegex(
            ValueError,
            "When providing `x` as a torch DataLoader, `sample_weights`",
        ):
            _ = epoch_iterator.EpochIterator(x=x, sample_weight=sample_weights)

    def test_python_generator_input(self):
        def generator_example():
            for i in range(100):
                yield (np.array([i]), np.array([i * 2]))

        x = generator_example()
        epoch_iter = epoch_iterator.EpochIterator(x=x)
        self.assertIsInstance(
            epoch_iter.data_adapter,
            data_adapters.GeneratorDataAdapter,
        )

    def test_unrecognized_data_type(self):
        x = "unsupported_data"
        with self.assertRaisesRegex(ValueError, "Unrecognized data type"):
            _ = epoch_iterator.EpochIterator(x=x)

    @parameterized.named_parameters(
        [
            {"testcase_name": "infinite", "infinite": True},
            {"testcase_name": "finite", "infinite": False},
        ]
    )
    def test_epoch_callbacks(self, infinite):
        class TestPyDataset(data_adapters.py_dataset_adapter.PyDataset):
            def __init__(
                self,
                workers=1,
                use_multiprocessing=False,
                max_queue_size=10,
                infinite=False,
            ):
                super().__init__(workers, use_multiprocessing, max_queue_size)
                self.data = np.random.rand(64, 2)
                self.batch_size = 16
                self.infinite = infinite

                # check that callbacks are called in the correct order
                self.tracker = []

            @property
            def num_batches(self):
                if self.infinite:
                    return None
                return len(self.data) // self.batch_size

            def on_epoch_begin(self):
                self.tracker.append(1)

            def __getitem__(self, index):
                idx = index % 2
                return self.data[
                    idx * self.batch_size : (idx + 1) * self.batch_size
                ]

            def on_epoch_end(self):
                self.tracker.append(2)

        ds = TestPyDataset(infinite=infinite)
        epoch_iter = epoch_iterator.EpochIterator(x=ds, steps_per_epoch=10)

        num_epochs = 5
        for epoch in range(num_epochs):
            for step, batch in epoch_iter.enumerate_epoch():
                pass

        self.assertAllEqual(ds.tracker, [1, 2] * num_epochs)
