import paddle

from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import is_tensor


def rgb_to_grayscale(image, data_format="channels_last"):
    raise NotImplementedError(
        "`rgb_to_grayscale` is not supported with paddle backend"
    )


def resize(
    image,
    size,
    interpolation="bilinear",
    antialias=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    fill_mode="constant",
    fill_value=0.0,
    data_format="channels_last",
):
    raise NotImplementedError(
        "`resize` is not supported with paddle backend"
    )


def affine_transform(
    image,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format="channels_last",
):
    raise NotImplementedError(
        "`affine_transform` is not supported with paddle backend"
    )


def map_coordinates(
    input, coordinates, order, fill_mode="constant", fill_value=0.0
):
    raise NotImplementedError(
        "`map_coordinates` is not supported with paddle backend"
    )
