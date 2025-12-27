import io
from unittest.mock import patch

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

    def test_progbar_pinned_output(self):
        """Verify that the actual ANSI codes are printed when pinned=True."""
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            target = 5
            pb = progbar.Progbar(target=target, pinned=True, interval=-1)
            for i in range(target):
                pb.update(i + 1)

            output = fake_out.getvalue()

            self.assertIn("\033[s", output)
            self.assertTrue("\033[1;1H" in output or "\033[2;1H" in output)
            self.assertIn("\033[K", output)
            self.assertIn("\033[u", output)

    def test_progbar_pinned_attribute(self):
        """Simple check for attribute assignment."""
        pb = progbar.Progbar(target=10, pinned=True)
        self.assertTrue(pb.pinned)
