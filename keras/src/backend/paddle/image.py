import numpy as np
import paddle
import paddle.nn.functional as F

from keras.src.backend.common.variables import standardize_dtype
from keras.src.backend.config import standardize_data_format
from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import to_paddle_dtype

RESIZE_INTERPOLATIONS = ("bilinear", "nearest", "bicubic")
UNSUPPORTED_INTERPOLATIONS = (
    "lanczos3",
    "lanczos5",
)


def _dtype_limits(dtype):
    if dtype == "bool":
        return 0, 1
    info = np.iinfo(dtype)
    return int(info.min), int(info.max)


def _resize_nearest(image, size):
    """Resize with nearest neighbors sampled at the output pixel centers.

    `F.interpolate(mode="nearest")` samples at the top-left corner of every
    output pixel, while the reference implementations sample at its center.
    """
    src_height, src_width = image.shape[-2], image.shape[-1]
    for axis, (src, dst) in enumerate(
        ((src_height, size[0]), (src_width, size[1])), start=2
    ):
        indices = paddle.arange(dst, dtype="float64")
        indices = paddle.floor((indices + 0.5) * (src / dst))
        indices = paddle.clip(indices, 0, src - 1).cast("int64")
        image = paddle.index_select(image, indices, axis=axis)
    return image


def rgb_to_grayscale(images, data_format=None):
    data_format = standardize_data_format(data_format)
    images = convert_to_tensor(images)
    if images.ndim not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    channel_axis = -1 if data_format == "channels_last" else -3
    if images.shape[channel_axis] not in (1, 3):
        raise ValueError(
            "Invalid channel size: expected 3 (RGB) or 1 (Grayscale). "
            f"Received input with shape: images.shape={images.shape}"
        )
    if images.shape[channel_axis] == 3:
        # `multiply` / `add` have no float16 or bfloat16 CPU kernel, and
        # integer inputs have to be weighted in floating point anyway.
        orig_dtype = images.dtype
        r, g, b = paddle.unbind(images.cast("float32"), axis=channel_axis)
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.unsqueeze(channel_axis).cast(orig_dtype)
    return images.clone()


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
    data_format = standardize_data_format(data_format)
    if interpolation in UNSUPPORTED_INTERPOLATIONS:
        raise ValueError(
            "Resizing with Lanczos interpolation is "
            "not supported by the paddle backend. "
            f"Received: interpolation={interpolation}."
        )
    if interpolation not in RESIZE_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{RESIZE_INTERPOLATIONS}. Received: interpolation={interpolation}"
        )
    if fill_mode != "constant":
        raise ValueError(
            "Invalid value for argument `fill_mode`. Only `'constant'` "
            f"is supported. Received: fill_mode={fill_mode}"
        )
    if pad_to_aspect_ratio and crop_to_aspect_ratio:
        raise ValueError(
            "Only one of `pad_to_aspect_ratio` & `crop_to_aspect_ratio` "
            "can be `True`."
        )
    if not len(size) == 2:
        raise ValueError(
            "Argument `size` must be a tuple of two elements "
            f"(height, width). Received: size={size}"
        )
    size = tuple(size)
    image = convert_to_tensor(image)
    out_dtype = standardize_dtype(image.dtype)
    if out_dtype not in ("float32", "float64"):
        image = image.cast("float32")
    if image.ndim not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={image.shape}"
        )
    has_batch = image.ndim == 4
    if not has_batch:
        image = paddle.unsqueeze(image, axis=0)

    if data_format == "channels_last":
        image = paddle.transpose(image, [0, 3, 1, 2])

    if crop_to_aspect_ratio:
        shape = image.shape
        height, width = shape[-2], shape[-1]
        target_height, target_width = size
        crop_height = int(float(width * target_height) / target_width)
        crop_height = max(min(height, crop_height), 1)
        crop_width = int(float(height * target_width) / target_height)
        crop_width = max(min(width, crop_width), 1)
        crop_box_hstart = int(float(height - crop_height) / 2)
        crop_box_wstart = int(float(width - crop_width) / 2)
        image = image[
            :,
            :,
            crop_box_hstart : crop_box_hstart + crop_height,
            crop_box_wstart : crop_box_wstart + crop_width,
        ]
    elif pad_to_aspect_ratio:
        shape = image.shape
        height, width = shape[-2], shape[-1]
        target_height, target_width = size
        pad_height = int(float(width * target_height) / target_width)
        pad_height = max(height, pad_height)
        pad_width = int(float(height * target_width) / target_height)
        pad_width = max(width, pad_width)
        img_box_hstart = int(float(pad_height - height) / 2)
        img_box_wstart = int(float(pad_width - width) / 2)

        batch_size = image.shape[0]
        channels = image.shape[1]
        if img_box_hstart > 0:
            padded_img = paddle.concat(
                [
                    paddle.full(
                        [batch_size, channels, img_box_hstart, width],
                        fill_value,
                        dtype=image.dtype,
                    ),
                    image,
                    paddle.full(
                        [batch_size, channels, img_box_hstart, width],
                        fill_value,
                        dtype=image.dtype,
                    ),
                ],
                axis=2,
            )
        else:
            padded_img = image
        if img_box_wstart > 0:
            padded_img = paddle.concat(
                [
                    paddle.full(
                        [batch_size, channels, height, img_box_wstart],
                        fill_value,
                        dtype=image.dtype,
                    ),
                    padded_img,
                    paddle.full(
                        [batch_size, channels, height, img_box_wstart],
                        fill_value,
                        dtype=image.dtype,
                    ),
                ],
                axis=3,
            )
        image = padded_img

    if antialias and interpolation not in ("bilinear", "bicubic"):
        # Paddle only supports antialiasing for bilinear and bicubic modes.
        # The parameter is irrelevant for the other modes.
        antialias = False
    if interpolation == "nearest":
        out = _resize_nearest(image, size)
    else:
        out = F.interpolate(
            image,
            size=size,
            mode=interpolation,
            align_corners=False,
            antialias=antialias,
        )

    if data_format == "channels_last":
        out = paddle.transpose(out, [0, 2, 3, 1])

    if not has_batch:
        out = paddle.squeeze(out, axis=0)

    if standardize_dtype(out.dtype) != out_dtype:
        if "int" in out_dtype or out_dtype == "bool":
            # Rounding before the cast avoids truncating e.g. 0.999 to 0.
            out = paddle.round(out)
            out = paddle.clip(out, *_dtype_limits(out_dtype))
        out = out.cast(to_paddle_dtype(out_dtype))
    return out


def affine_transform(
    images,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format=None,
):
    raise NotImplementedError(
        "`affine_transform` is not supported with paddle backend"
    )


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0.0
):
    raise NotImplementedError(
        "`map_coordinates` is not supported with paddle backend"
    )


def rgb_to_hsv(images, data_format=None):
    raise NotImplementedError(
        "`rgb_to_hsv` is not supported with paddle backend"
    )


def hsv_to_rgb(images, data_format=None):
    raise NotImplementedError(
        "`hsv_to_rgb` is not supported with paddle backend"
    )


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    raise NotImplementedError(
        "`perspective_transform` is not supported with paddle backend"
    )


def compute_homography_matrix(start_points, end_points):
    raise NotImplementedError(
        "`compute_homography_matrix` is not supported with paddle backend"
    )


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    raise NotImplementedError(
        "`gaussian_blur` is not supported with paddle backend"
    )


def elastic_transform(
    images,
    alpha=20.0,
    sigma=5.0,
    interpolation="bilinear",
    fill_mode="reflect",
    fill_value=0.0,
    seed=None,
    data_format=None,
):
    raise NotImplementedError(
        "`elastic_transform` is not supported with paddle backend"
    )


def scale_and_translate(
    images,
    output_shape,
    scale,
    translation,
    spatial_dims,
    method,
    antialias=True,
):
    raise NotImplementedError(
        "`scale_and_translate` is not supported with paddle backend"
    )


def sobel_edges(images, data_format=None):
    raise NotImplementedError(
        "`sobel_edges` is not supported with paddle backend"
    )
