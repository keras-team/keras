import numpy as np
import pytest
import tensorflow as tf

from keras_core import backend
from keras_core import testing
from keras_core.trainers import epoch_iterator


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
        self.assertTrue(iterator._insufficient_data)

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
