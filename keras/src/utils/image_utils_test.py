import os

import numpy as np
from absl.testing import parameterized

from keras.src import testing
from keras.src.utils import img_to_array
from keras.src.utils import load_img
from keras.src.utils import save_img


class SaveImgTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("rgb_explicit_format", (50, 50, 3), "rgb.jpg", "jpg", True),
        ("rgba_explicit_format", (50, 50, 4), "rgba.jpg", "jpg", True),
        ("rgb_inferred_format", (50, 50, 3), "rgb_inferred.jpg", None, False),
        ("rgba_inferred_format", (50, 50, 4), "rgba_inferred.jpg", None, False),
    )
    def test_save_jpg(self, shape, name, file_format, use_explicit_format):
        tmp_dir = self.get_temp_dir()
        path = os.path.join(tmp_dir, name)

        img = np.random.randint(0, 256, size=shape, dtype=np.uint8)

        # Test the actual inferred case - don't pass file_format at all
        if use_explicit_format:
            save_img(path, img, file_format=file_format)
        else:
            save_img(path, img)  # Let it infer from path

        self.assertTrue(os.path.exists(path))

        # Verify saved image is correctly converted to RGB if needed
        loaded_img = load_img(path)
        loaded_array = img_to_array(loaded_img)
        self.assertEqual(loaded_array.shape, (50, 50, 3))
