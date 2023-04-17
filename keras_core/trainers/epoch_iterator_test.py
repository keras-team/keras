import numpy as np
import tensorflow as tf

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
            steps_seen.append(step)
            self.assertEqual(len(batch), 3)
            if return_type == "np":
                self.assertTrue(isinstance(batch[0], np.ndarray))
            else:
                self.assertTrue(isinstance(batch[0], tf.Tensor))
        self.assertEqual(steps_seen, [0, 1, 2, 3, 4, 5, 6])

    def test_basic_flow_np(self):
        self._test_basic_flow("np")

    def test_basic_flow_tf(self):
        self._test_basic_flow("tf")
