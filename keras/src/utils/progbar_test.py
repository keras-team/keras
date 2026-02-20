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

    def test_stateful_metrics_displayed_as_is(self):
        """Stateful metrics should be displayed without additional averaging."""
        pb = progbar.Progbar(
            target=3, verbose=2, stateful_metrics=["acc"]
        )
        # Simulate 3 steps where the stateful metric "acc" grows.
        pb.update(1, values=[("acc", 0.5)])
        pb.update(2, values=[("acc", 0.6)])
        pb.update(3, values=[("acc", 0.7)], finalize=True)
        # After final update the stored value should be the last one, not an
        # average of previous updates.
        stored_value = pb._values["acc"][0] / pb._values["acc"][1]
        self.assertAlmostEqual(stored_value, 0.7)

    def test_non_stateful_metrics_are_averaged(self):
        """Non-stateful metrics should be averaged across updates."""
        pb = progbar.Progbar(target=3, verbose=2)
        pb.update(1, values=[("loss", 1.0)])
        pb.update(2, values=[("loss", 2.0)])
        pb.update(3, values=[("loss", 3.0)])
        # Average of 1.0, 2.0, 3.0 = 2.0
        stored_value = pb._values["loss"][0] / pb._values["loss"][1]
        self.assertAlmostEqual(stored_value, 2.0)
