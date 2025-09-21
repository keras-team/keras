import os
import numpy as np
import pytest
from keras.utils import save_img, load_img, img_to_array


@pytest.mark.parametrize(
    "shape, filename",
    [
        ((50, 50, 3), "rgb.jpg"),
        ((50, 50, 4), "rgba.jpg"),
        ((50, 50, 3), "rgb.jpeg"),
        ((50, 50, 4), "rgba.jpeg"),
    ],
)
def test_save_img_jpg_and_jpeg(tmp_path, shape, filename):
    # Create random RGB or RGBA image
    img = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    path = tmp_path / filename

    # Save using explicit format
    save_img(path, img, file_format="jpg")
    assert os.path.exists(path)

    # Load back and check shape (RGBA → RGB if JPEG)
    loaded_img = load_img(path)
    loaded_array = img_to_array(loaded_img)

    # Always 3 channels after save (JPEG does not support RGBA)
    assert loaded_array.shape == (50, 50, 3)
    assert loaded_array.dtype == np.float32  # keras.load_img returns float32
