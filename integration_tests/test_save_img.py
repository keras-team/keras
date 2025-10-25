import os

import numpy as np
import pytest

from keras.utils import img_to_array
from keras.utils import load_img
from keras.utils import save_img


@pytest.mark.parametrize(
    "shape, name",
    [
        ((50, 50, 3), "rgb.jpg"),
        ((50, 50, 4), "rgba.jpg"),
    ],
)
def test_save_jpg(tmp_path, shape, name):
    img = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    path = tmp_path / name
    save_img(path, img, file_format="jpg")
    assert os.path.exists(path)

    # Check that the image was saved correctly and converted to RGB if needed.
    loaded_img = load_img(path)
    loaded_array = img_to_array(loaded_img)
    assert loaded_array.shape == (50, 50, 3)
