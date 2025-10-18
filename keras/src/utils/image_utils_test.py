import os
import numpy as np
from absl.testing import parameterized
from keras.src import testing
from keras.utils import img_to_array, load_img, save_img


class SaveImgJpgTest(testing.TestCase, parameterized.TestCase):

    @parameterized.parameters(
        ((50, 50, 3), "rgb.jpg"),
        ((50, 50, 4), "rgba.jpg"),
    )
    def test_save_jpg(self, shape, name):
        tmp_dir = self.get_temp_dir()
        path = os.path.join(tmp_dir, name)

        img = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        save_img(path, img, file_format="jpg")
        self.assertTrue(os.path.exists(path))

        # Check that the image was saved correctly
        # and converted to RGB if needed.
        loaded_img = load_img(path)
        loaded_array = img_to_array(loaded_img)
        self.assertEqual(loaded_array.shape, (50, 50, 3))
