import numpy as np
from absl.testing import parameterized

from keras.src import testing
from keras.src.utils import progbar


class ProgbarTest(testing.TestCase):
    @parameterized.named_parameters(
        [
            ("float", "float"),
            ("np", "np"),
            ("list", "list"),
        ]
    )
    def test_update(self, value_type):
        if value_type == "float":
            values = 1.0
        elif value_type == "np":
            values = np.array(1.0)
        elif value_type == "list":
            values = [0.0, 1.0, 2.0]
        else:
            raise ValueError("Unknown value_type")
        pb = progbar.Progbar(target=1, verbose=1)

        pb.update(1, values=[("values", values)], finalize=True)

    @parameterized.named_parameters(
        [
            ("verbose_1", 1),
            ("verbose_2", 2),
        ]
    )
    def test_zero_target(self, verbose):
        pb = progbar.Progbar(target=0, verbose=verbose)
        pb.update(0, finalize=True)
