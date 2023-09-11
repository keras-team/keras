import os

import numpy as np
import pytest

from keras_core import layers
from keras_core import models
from keras_core import testing
from keras_core.utils import summary_utils


class SummaryUtilsTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_print_model_summary(self):
        inputs = layers.Input((2,))
        outputs = layers.Dense(3)(inputs)
        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse", metrics=["mse"])
        # Trigger the optimizer weights creation
        model.fit(x=np.zeros([4, 2]), y=np.zeros([4, 3]))

        file_name = "model_1.txt"
        temp_dir = self.get_temp_dir()
        fpath = os.path.join(temp_dir, file_name)
        writer = open(fpath, "w")

        def print_to_file(text, line_break=False):
            print(text, file=writer)

        try:
            summary_utils.print_summary(model, print_fn=print_to_file)
            writer.close()
            self.assertTrue(os.path.exists(fpath))
            with open(fpath, "r") as reader:
                summary_content = reader.read()
            # self.assertEqual(len(lines), 15)
            self.assertIn("Total params: 29", summary_content)
            self.assertIn("Trainable params: 9", summary_content)
            self.assertIn("Non-trainable params: 0", summary_content)
            self.assertIn("Optimizer params: 20", summary_content)
        except ImportError:
            pass
