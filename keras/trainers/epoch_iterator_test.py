import numpy as np
import pytest
import tensorflow as tf

from keras import backend
from keras import testing
from keras.trainers import data_adapters
from keras.trainers import epoch_iterator


class TestEpochIterator(testing.TestCase):
    def _test_basic_flow(self, return_type):
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
        for step, batch in iterator.enumerate_epoch(return_type=return_type):
            batch = batch[0]
            steps_seen.append(step)
            self.assertEqual(len(batch), 3)
            if return_type == "np":
                self.assertIsInstance(batch[0], np.ndarray)
            else:
                self.assertIsInstance(batch[0], tf.Tensor)
        self.assertEqual(steps_seen, [0, 1, 2, 3, 4, 5, 6])

    def test_basic_flow_np(self):
        self._test_basic_flow("np")

    def test_basic_flow_tf(self):
        self._test_basic_flow("tf")

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
        for _, batch in iterator.enumerate_epoch(return_type="np"):
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

    def test_invalid_return_type_in_get_iterator(self):
        x = np.random.random((100, 16))
        y = np.random.random((100, 4))
        epoch_iter = epoch_iterator.EpochIterator(x=x, y=y)

        with self.assertRaisesRegex(
            ValueError,
            "Argument `return_type` must be one of `{'np', 'tf', 'auto'}`",
        ):
            _ = epoch_iter._get_iterator("unsupported")
