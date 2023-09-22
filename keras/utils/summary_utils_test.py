import numpy as np
import pytest
from absl.testing import parameterized

from keras import layers
from keras import models
from keras import testing
from keras.utils import summary_utils


class SummaryUtilsTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters([("adam",), (None,)])
    @pytest.mark.requires_trainable_backend
    def test_print_model_summary(self, optimizer):
        inputs = layers.Input((2,))
        outputs = layers.Dense(3)(inputs)
        model = models.Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
        if optimizer:
            # Trigger the optimizer weights creation
            model.fit(x=np.zeros([4, 2]), y=np.zeros([4, 3]))

        summary_content = []

        def print_to_variable(text, line_break=False):
            summary_content.append(text)

        try:
            summary_utils.print_summary(model, print_fn=print_to_variable)
            summary_content = "\n".join(summary_content)
            if optimizer:
                self.assertIn("Total params: 29", summary_content)
                self.assertIn("Trainable params: 9", summary_content)
                self.assertIn("Non-trainable params: 0", summary_content)
                self.assertIn("Optimizer params: 20", summary_content)
            else:
                self.assertIn("Total params: 9", summary_content)
                self.assertIn("Trainable params: 9", summary_content)
                self.assertIn("Non-trainable params: 0", summary_content)
                self.assertNotIn("Optimizer params", summary_content)
        except ImportError:
            pass
