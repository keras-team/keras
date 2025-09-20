import numpy as np
import os
from keras.utils import save_img

def test_save_jpg_rgb(tmp_path):
    img = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    path = tmp_path / "rgb.jpg"
    save_img(path, img, file_format="jpg")
    assert os.path.exists(path)

def test_save_jpg_rgba(tmp_path):
    img = np.random.randint(0, 256, size=(50, 50, 4), dtype=np.uint8)
    path = tmp_path / "rgba.jpg"
    save_img(path, img, file_format="jpg")
    assert os.path.exists(path)
