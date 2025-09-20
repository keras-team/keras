import numpy as np
import os
import pytest
from keras.utils import save_img


def test_save_jpg_rgb_with_format(tmp_path):
    """Saving RGB image with explicit file_format='jpg'."""
    img = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    path = tmp_path / "rgb_explicit.jpg"
    save_img(path, img, file_format="jpg")
    assert os.path.exists(path)


def test_save_jpg_rgb_infer_from_extension(tmp_path):
    """Saving RGB image where format is inferred from .jpg extension."""
    img = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    path = tmp_path / "rgb_infer.jpg"
    save_img(path, img)  # no file_format passed
    assert os.path.exists(path)


def test_save_jpg_rgba_with_format(tmp_path):
    """Saving RGBA image with explicit file_format='jpg' (should auto-convert)."""
    img = np.random.randint(0, 256, size=(50, 50, 4), dtype=np.uint8)
    path = tmp_path / "rgba_explicit.jpg"
    save_img(path, img, file_format="jpg")
    assert os.path.exists(path)


def test_save_jpg_rgba_infer_from_extension(tmp_path):
    """Saving RGBA image where format is inferred from .jpg extension (should auto-convert)."""
    img = np.random.randint(0, 256, size=(50, 50, 4), dtype=np.uint8)
    path = tmp_path / "rgba_infer.jpg"
    save_img(path, img)  # no file_format passed
    assert os.path.exists(path)
