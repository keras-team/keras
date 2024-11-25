import jax
import ml_dtypes
import numpy as np

from keras.src import backend
from keras.src.backend.numpy.core import convert_to_tensor
from keras.src.utils.module_utils import scipy

RESIZE_INTERPOLATIONS = (
    "bilinear",
    "nearest",
    "lanczos3",
    "lanczos5",
    "bicubic",
)


def rgb_to_grayscale(images, data_format=None):
    images = convert_to_tensor(images)
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    # Convert to floats
    original_dtype = images.dtype
    compute_dtype = backend.result_type(images.dtype, float)
    images = images.astype(compute_dtype)

    # Ref: tf.image.rgb_to_grayscale
    rgb_weights = np.array([0.2989, 0.5870, 0.1140], dtype=images.dtype)
    grayscales = np.tensordot(images, rgb_weights, axes=(channels_axis, -1))
    grayscales = np.expand_dims(grayscales, axis=channels_axis)
    return grayscales.astype(original_dtype)


def rgb_to_hsv(images, data_format=None):
    # Ref: dm_pix
    images = convert_to_tensor(images)
    dtype = backend.standardize_dtype(images.dtype)
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if not backend.is_float_dtype(dtype):
        raise ValueError(
            "Invalid images dtype: expected float dtype. "
            f"Received: images.dtype={dtype}"
        )
    eps = ml_dtypes.finfo(dtype).eps
    images = np.where(np.abs(images) < eps, 0.0, images)
    red, green, blue = np.split(images, 3, channels_axis)
    red = np.squeeze(red, channels_axis)
    green = np.squeeze(green, channels_axis)
    blue = np.squeeze(blue, channels_axis)

    def rgb_planes_to_hsv_planes(r, g, b):
        value = np.maximum(np.maximum(r, g), b)
        minimum = np.minimum(np.minimum(r, g), b)
        range_ = value - minimum

        safe_value = np.where(value > 0, value, 1.0)
        safe_range = np.where(range_ > 0, range_, 1.0)

        saturation = np.where(value > 0, range_ / safe_value, 0.0)
        norm = 1.0 / (6.0 * safe_range)

        hue = np.where(
            value == g,
            norm * (b - r) + 2.0 / 6.0,
            norm * (r - g) + 4.0 / 6.0,
        )
        hue = np.where(value == r, norm * (g - b), hue)
        hue = np.where(range_ > 0, hue, 0.0) + (hue < 0.0).astype(hue.dtype)
        return hue, saturation, value

    images = np.stack(
        rgb_planes_to_hsv_planes(red, green, blue), axis=channels_axis
    )
    return images.astype(dtype)


def hsv_to_rgb(images, data_format=None):
    # Ref: dm_pix
    images = convert_to_tensor(images)
    dtype = images.dtype
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if not backend.is_float_dtype(dtype):
        raise ValueError(
            "Invalid images dtype: expected float dtype. "
            f"Received: images.dtype={backend.standardize_dtype(dtype)}"
        )
    hue, saturation, value = np.split(images, 3, channels_axis)
    hue = np.squeeze(hue, channels_axis)
    saturation = np.squeeze(saturation, channels_axis)
    value = np.squeeze(value, channels_axis)

    def hsv_planes_to_rgb_planes(hue, saturation, value):
        dh = np.mod(hue, 1.0) * 6.0
        dr = np.clip(np.abs(dh - 3.0) - 1.0, 0.0, 1.0)
        dg = np.clip(2.0 - np.abs(dh - 2.0), 0.0, 1.0)
        db = np.clip(2.0 - np.abs(dh - 4.0), 0.0, 1.0)
        one_minus_s = 1.0 - saturation

        red = value * (one_minus_s + saturation * dr)
        green = value * (one_minus_s + saturation * dg)
        blue = value * (one_minus_s + saturation * db)
        return red, green, blue

    images = np.stack(
        hsv_planes_to_rgb_planes(hue, saturation, value), axis=channels_axis
    )
    return images.astype(dtype)


def resize(
    images,
    size,
    interpolation="bilinear",
    antialias=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    fill_mode="constant",
    fill_value=0.0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
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
    target_height, target_width = size
    if len(images.shape) == 4:
        if data_format == "channels_last":
            size = (images.shape[0],) + size + (images.shape[-1],)
        else:
            size = (images.shape[0], images.shape[1]) + size
    elif len(images.shape) == 3:
        if data_format == "channels_last":
            size = size + (images.shape[-1],)
        else:
            size = (images.shape[0],) + size
    else:
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    if crop_to_aspect_ratio:
        shape = images.shape
        if data_format == "channels_last":
            height, width = shape[-3], shape[-2]
        else:
            height, width = shape[-2], shape[-1]
        crop_height = int(float(width * target_height) / target_width)
        crop_height = max(min(height, crop_height), 1)
        crop_width = int(float(height * target_width) / target_height)
        crop_width = max(min(width, crop_width), 1)
        crop_box_hstart = int(float(height - crop_height) / 2)
        crop_box_wstart = int(float(width - crop_width) / 2)
        if data_format == "channels_last":
            if len(images.shape) == 4:
                images = images[
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                    :,
                ]
            else:
                images = images[
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                    :,
                ]
        else:
            if len(images.shape) == 4:
                images = images[
                    :,
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                ]
            else:
                images = images[
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                ]
    elif pad_to_aspect_ratio:
        shape = images.shape
        batch_size = images.shape[0]
        if data_format == "channels_last":
            height, width, channels = shape[-3], shape[-2], shape[-1]
        else:
            channels, height, width = shape[-3], shape[-2], shape[-1]
        pad_height = int(float(width * target_height) / target_width)
        pad_height = max(height, pad_height)
        pad_width = int(float(height * target_width) / target_height)
        pad_width = max(width, pad_width)
        img_box_hstart = int(float(pad_height - height) / 2)
        img_box_wstart = int(float(pad_width - width) / 2)
        if data_format == "channels_last":
            if len(images.shape) == 4:
                padded_img = (
                    np.ones(
                        (
                            batch_size,
                            pad_height + height,
                            pad_width + width,
                            channels,
                        ),
                        dtype=images.dtype,
                    )
                    * fill_value
                )
                padded_img[
                    :,
                    img_box_hstart : img_box_hstart + height,
                    img_box_wstart : img_box_wstart + width,
                    :,
                ] = images
            else:
                padded_img = (
                    np.ones(
                        (pad_height + height, pad_width + width, channels),
                        dtype=images.dtype,
                    )
                    * fill_value
                )
                padded_img[
                    img_box_hstart : img_box_hstart + height,
                    img_box_wstart : img_box_wstart + width,
                    :,
                ] = images
        else:
            if len(images.shape) == 4:
                padded_img = (
                    np.ones(
                        (
                            batch_size,
                            channels,
                            pad_height + height,
                            pad_width + width,
                        ),
                        dtype=images.dtype,
                    )
                    * fill_value
                )
                padded_img[
                    :,
                    :,
                    img_box_hstart : img_box_hstart + height,
                    img_box_wstart : img_box_wstart + width,
                ] = images
            else:
                padded_img = (
                    np.ones(
                        (channels, pad_height + height, pad_width + width),
                        dtype=images.dtype,
                    )
                    * fill_value
                )
                padded_img[
                    :,
                    img_box_hstart : img_box_hstart + height,
                    img_box_wstart : img_box_wstart + width,
                ] = images
        images = padded_img

    return np.array(
        jax.image.resize(
            images, size, method=interpolation, antialias=antialias
        )
    )


AFFINE_TRANSFORM_INTERPOLATIONS = {  # map to order
    "nearest": 0,
    "bilinear": 1,
}
AFFINE_TRANSFORM_FILL_MODES = {
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
}


def affine_transform(
    images,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS.keys():
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )
    if fill_mode not in AFFINE_TRANSFORM_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{AFFINE_TRANSFORM_FILL_MODES}. Received: fill_mode={fill_mode}"
        )

    transform = convert_to_tensor(transform)

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if len(transform.shape) not in (1, 2):
        raise ValueError(
            "Invalid transform rank: expected rank 1 (single transform) "
            "or rank 2 (batch of transforms). Received input with shape: "
            f"transform.shape={transform.shape}"
        )

    # scipy.ndimage.map_coordinates lacks support for half precision.
    input_dtype = images.dtype
    if input_dtype == "float16":
        images = images.astype("float32")

    # unbatched case
    need_squeeze = False
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        need_squeeze = True
    if len(transform.shape) == 1:
        transform = np.expand_dims(transform, axis=0)

    if data_format == "channels_first":
        images = np.transpose(images, (0, 2, 3, 1))

    batch_size = images.shape[0]

    # get indices
    meshgrid = np.meshgrid(
        *[np.arange(size) for size in images.shape[1:]], indexing="ij"
    )
    indices = np.concatenate(
        [np.expand_dims(x, axis=-1) for x in meshgrid], axis=-1
    )
    indices = np.tile(indices, (batch_size, 1, 1, 1, 1))

    # swap the values
    a0 = transform[:, 0].copy()
    a2 = transform[:, 2].copy()
    b1 = transform[:, 4].copy()
    b2 = transform[:, 5].copy()
    transform[:, 0] = b1
    transform[:, 2] = b2
    transform[:, 4] = a0
    transform[:, 5] = a2

    # deal with transform
    transform = np.pad(transform, pad_width=[[0, 0], [0, 1]], constant_values=1)
    transform = np.reshape(transform, (batch_size, 3, 3))
    offset = transform[:, 0:2, 2].copy()
    offset = np.pad(offset, pad_width=[[0, 0], [0, 1]])
    transform[:, 0:2, 2] = 0

    # transform the indices
    coordinates = np.einsum("Bhwij, Bjk -> Bhwik", indices, transform)
    coordinates = np.moveaxis(coordinates, source=-1, destination=1)
    coordinates += np.reshape(a=offset, newshape=(*offset.shape, 1, 1, 1))

    # apply affine transformation
    affined = np.stack(
        [
            map_coordinates(
                images[i],
                coordinates[i],
                order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                fill_mode=fill_mode,
                fill_value=fill_value,
            )
            for i in range(batch_size)
        ],
        axis=0,
    )

    if data_format == "channels_first":
        affined = np.transpose(affined, (0, 3, 1, 2))
    if need_squeeze:
        affined = np.squeeze(affined, axis=0)
    if input_dtype == "float16":
        affined = affined.astype(input_dtype)
    return affined


MAP_COORDINATES_FILL_MODES = {
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
}


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0.0
):
    inputs = convert_to_tensor(inputs)
    coordinates = convert_to_tensor(coordinates)
    if coordinates.shape[0] != len(inputs.shape):
        raise ValueError(
            "First dim of `coordinates` must be the same as the rank of "
            "`inputs`. "
            f"Received inputs with shape: {inputs.shape} and coordinate "
            f"leading dim of {coordinates.shape[0]}"
        )
    if len(coordinates.shape) < 2:
        raise ValueError(
            "Invalid coordinates rank: expected at least rank 2."
            f" Received input with shape: {coordinates.shape}"
        )
    if fill_mode not in MAP_COORDINATES_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected one of "
            f"{set(MAP_COORDINATES_FILL_MODES.keys())}. Received: "
            f"fill_mode={fill_mode}"
        )
    if order not in range(2):
        raise ValueError(
            "Invalid value for argument `order`. Expected one of "
            f"{[0, 1]}. Received: order={order}"
        )
    # SciPy's implementation of map_coordinates handles boundaries incorrectly,
    # unless mode='reflect'. For order=1, this only affects interpolation
    # outside the bounds of the original array.
    # https://github.com/scipy/scipy/issues/2640
    padding = [
        (
            max(-np.floor(c.min()).astype(int) + 1, 0),
            max(np.ceil(c.max()).astype(int) + 1 - size, 0),
        )
        for c, size in zip(coordinates, inputs.shape)
    ]
    shifted_coords = [c + p[0] for p, c in zip(padding, coordinates)]
    pad_mode = {
        "nearest": "edge",
        "mirror": "reflect",
        "reflect": "symmetric",
    }.get(fill_mode, fill_mode)
    if fill_mode == "constant":
        padded = np.pad(
            inputs, padding, mode=pad_mode, constant_values=fill_value
        )
    else:
        padded = np.pad(inputs, padding, mode=pad_mode)
    result = scipy.ndimage.map_coordinates(
        padded, shifted_coords, order=order, mode=fill_mode, cval=fill_value
    )
    return result
