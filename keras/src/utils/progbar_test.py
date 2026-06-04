import contextlib
import io

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

    def test_stateful_stateless_raises(self):
        with self.assertRaisesRegex(ValueError, "Only one of"):
            progbar.Progbar(target=1, stateful_metrics=[], stateless_metrics=[])

    @parameterized.named_parameters(
        [
            ("all_stateless", [], None, "1."),
            ("one_stateless", ["value"], None, "2."),
            ("all_stateful", None, [], "2."),
            ("one_stateful", None, ["value"], "1."),
        ]
    )
    def test_stateful_stateless_output(
        self, stateful_metrics, stateless_metrics, expected_output
    ):
        captured = io.StringIO()
        pb = progbar.Progbar(
            target=3,
            stateful_metrics=stateful_metrics,
            stateless_metrics=stateless_metrics,
            interval=0.0,
            verbose=1,
        )

        pb.update(1, values=[("value", 0.0)])
        with contextlib.redirect_stdout(captured):
            pb.update(2, values=[("value", 2.0)])

        self.assertIn(f"value: {expected_output}", captured.getvalue())
