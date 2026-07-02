import paddle
import paddle.nn.functional as F

from keras.src.backend.paddle.core import convert_to_tensor


def rgb_to_grayscale(image, data_format="channels_last"):
    image = convert_to_tensor(image, "float32")
    if data_format == "channels_last":
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
    else:
        r, g, b = image[:, 0], image[:, 1], image[:, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    if data_format == "channels_last":
        return gray.unsqueeze(-1)
    return gray.unsqueeze(1)


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
    image = convert_to_tensor(image, "float32")
    has_batch = image.ndim == 4
    if not has_batch:
        image = paddle.unsqueeze(image, axis=0)

    if data_format == "channels_last":
        image = paddle.transpose(image, [0, 3, 1, 2])

    if interpolation == "bilinear":
        mode = "bilinear"
    elif interpolation == "nearest":
        mode = "nearest"
    elif interpolation == "bicubic":
        mode = "bicubic"
    else:
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    out = F.interpolate(image, size=size, mode=mode, align_corners=False)

    if data_format == "channels_last":
        out = paddle.transpose(out, [0, 2, 3, 1])

    if not has_batch:
        out = paddle.squeeze(out, axis=0)
    return out


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
