import functools
import itertools
import operator

import mlx.core as mx

from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import to_mlx_dtype


def rgb_to_grayscale(image, data_format="channels_last"):
    image = convert_to_tensor(image)
    if data_format == "channels_first":
        if len(image.shape) == 4:
            image = mx.transpose(image, (0, 2, 3, 1))
        elif len(image.shape) == 3:
            image = mx.transpose(image, (1, 2, 0))
        else:
            raise ValueError(
                "Invalid input rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"image.shape={image.shape}"
            )
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    grayscale_image = 0.2989 * red + 0.5870 * green + 0.1140 * blue
    grayscale_image = mx.expand_dims(grayscale_image, axis=-1)
    if data_format == "channels_first":
        if len(image.shape) == 4:
            grayscale_image = mx.transpose(grayscale_image, (0, 3, 1, 2))
        elif len(image.shape) == 3:
            grayscale_image = mx.transpose(grayscale_image, (2, 0, 1))
    return mx.array(grayscale_image)


def _mirror_index_fixer(index, size):
    s = size - 1  # Half-wavelength of triangular wave
    # Scaled, integer-valued version of the triangular wave |x - round(x)|
    return mx.abs((index + s) % (2 * s) - s)


def _reflect_index_fixer(index, size):
    return mx.floor_divide(
        _mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2
    )


_INDEX_FIXERS = {
    # we need to take care of out-of-bound indices in torch
    "constant": lambda index, size: mx.clip(index, 0, size - 1),
    "nearest": lambda index, size: mx.clip(index, 0, size - 1),
    "wrap": lambda index, size: index % size,
    "mirror": _mirror_index_fixer,
    "reflect": _reflect_index_fixer,
}


def _is_integer(a):
    # Should we add bool?
    return to_mlx_dtype(a.dtype) in (
        mx.int32,
        mx.uint32,
        mx.int64,
        mx.uint64,
        mx.int16,
        mx.uint16,
        mx.int8,
        mx.uint8,
    )


def _nearest_indices_and_weights(coordinate):
    coordinate = coordinate if _is_integer(coordinate) else mx.round(coordinate)
    index = coordinate.astype(mx.int32)
    return [(index, 1)]


def _linear_indices_and_weights(coordinate):
    lower = mx.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.astype(mx.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


def _extract_coordinates(
    src,
    coordinates,
    interpolation_function,
    index_fixer,
    fill_value=0,
    check_validity=True,
    start_axis=0,
):
    def _expand(x):
        if not isinstance(x, mx.array):
            return x
        return x.reshape(*x.shape, *([1] * (src.ndim - x.ndim)))

    indices = []
    for ci, size in zip(coordinates, src.shape[start_axis:]):
        indices.append(
            [
                (
                    index_fixer(index, size),
                    _expand(mx.logical_and((0 <= index), (index < size))),
                    _expand(weight),
                )
                for index, weight in interpolation_function(ci)
            ]
        )

    outputs = []
    empty_slices = (slice(None),) * start_axis
    for items in itertools.product(*indices):
        indices, validities, weights = zip(*items)
        index = empty_slices + indices
        contribution = src[index]

        # Check if we need to replace some with fill value
        if check_validity:
            all_valid = functools.reduce(operator.and_, validities)
            contribution = mx.where(all_valid, contribution, fill_value)

        # Multiply with the weight if it isn't 1.0
        weight = functools.reduce(operator.mul, weights)
        if not (isinstance(weight, (float, int)) and weight == 1):
            contribution = contribution * weight

        outputs.append(contribution)

    result = functools.reduce(operator.add, outputs)
    if _is_integer(src) and not _is_integer(result):
        result = mx.round(result)

    return result.astype(src.dtype)


def map_coordinates(
    input, coordinates, order, fill_mode="constant", fill_value=0.0
):
    input_arr = convert_to_tensor(input)
    coordinate_arrs = [convert_to_tensor(c) for c in coordinates]
    # skip tensor creation as possible
    if isinstance(fill_value, (int, float)) and _is_integer(input_arr):
        fill_value = int(fill_value)

    if len(coordinates) != len(input_arr.shape):
        raise ValueError(
            "coordinates must be a sequence of length input.shape, but "
            f"{len(coordinates)} != {len(input_arr.shape)}"
        )

    index_fixer = _INDEX_FIXERS.get(fill_mode)
    if index_fixer is None:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected one of "
            f"{set(_INDEX_FIXERS.keys())}. Received: fill_mode={fill_mode}"
        )

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    else:
        raise NotImplementedError("map_coordinates currently requires order<=1")

    return _extract_coordinates(
        src=input_arr,
        coordinates=coordinate_arrs,
        interpolation_function=interp_fun,
        index_fixer=index_fixer,
        fill_value=fill_value,
        check_validity=fill_mode == "constant",
        start_axis=0,
    )


AFFINE_TRANSFORM_INTERPOLATIONS = {
    "nearest": _nearest_indices_and_weights,
    "bilinear": _linear_indices_and_weights,
}
AFFINE_TRANSFORM_FILL_MODES = {
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
}


def _affine_transform(
    src,
    transform,
    target_size,
    interpolation_function,
    index_fixer,
    fill_value=0,
    check_validity=True,
):
    y_target = mx.arange(target_size[0]).reshape(1, -1, 1)
    x_target = mx.arange(target_size[1]).reshape(1, 1, -1)
    a0, a1, a2, b0, b1, b2, c0, c1 = [
        t.reshape(-1, 1, 1)
        for t in transform.T.reshape(-1).reshape(8, -1).split(8)
    ]
    # TODO: Should we ignore c0 and c1 as the docs say they are only used in
    #       the tf backend?
    k = c0 * x_target + c1 * y_target + 1
    x_src = (a0 * x_target + a1 * y_target + a2) / k
    y_src = (b0 * x_target + b1 * y_target + b2) / k

    # not batched
    if src.ndim == 3:
        indices = [y_src.squeeze(0), x_src.squeeze(0)]

    # batched
    else:
        indices = [
            mx.arange(len(src)).reshape(-1, 1, 1),
            y_src,
            x_src,
        ]

    return _extract_coordinates(
        src=src,
        coordinates=indices,
        interpolation_function=interpolation_function,
        index_fixer=index_fixer,
        fill_value=fill_value,
        check_validity=check_validity,
        start_axis=0,
    )


def affine_transform(
    image,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format="channels_last",
):
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

    image = convert_to_tensor(image)
    transform = convert_to_tensor(transform)

    if image.ndim not in (3, 4):
        raise ValueError(
            "Invalid image rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"image.shape={image.shape}"
        )
    if transform.ndim not in (1, 2):
        raise ValueError(
            "Invalid transform rank: expected rank 1 (single transform) "
            "or rank 2 (batch of transforms). Received input with shape: "
            f"transform.shape={transform.shape}"
        )

    if data_format == "channels_first":
        image = (
            image.transpose(0, 2, 3, 1)
            if image.ndim == 4
            else image.transpose(1, 2, 0)
        )

    result = _affine_transform(
        src=image,
        transform=transform,
        target_size=image.shape[:2] if image.ndim == 3 else image.shape[1:3],
        interpolation_function=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
        index_fixer=_INDEX_FIXERS[fill_mode],
        fill_value=fill_value,
        check_validity=fill_mode == "constant",
    )

    if data_format == "channels_first":
        result = (
            result.transpose(0, 3, 1, 2)
            if image.ndim == 4
            else result.transpose(2, 0, 1)
        )

    return result


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
    if antialias:
        raise NotImplementedError(
            "Antialiasing not implemented for the MLX backend"
        )
    if pad_to_aspect_ratio and crop_to_aspect_ratio:
        raise ValueError(
            "Only one of `pad_to_aspect_ratio` & `crop_to_aspect_ratio` "
            "can be `True`."
        )
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS.keys():
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )
    target_height, target_width = size
    size = tuple(size)
    image = convert_to_tensor(image)

    if image.ndim not in (3, 4):
        raise ValueError(
            "Invalid input rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"image.shape={image.shape}"
        )

    if crop_to_aspect_ratio:
        shape = image.shape
        if data_format == "channels_last":
            height, width = shape[-3], shape[-2]
        else:
            height, width = shape[-2], shape[-1]
        crop_height = int(float(width * target_height) / target_width)
        crop_height = min(height, crop_height)
        crop_width = int(float(height * target_width) / target_height)
        crop_width = min(width, crop_width)
        crop_box_hstart = int(float(height - crop_height) / 2)
        crop_box_wstart = int(float(width - crop_width) / 2)
        if data_format == "channels_last":
            if len(image.shape) == 4:
                image = image[
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                    :,
                ]
            else:
                image = image[
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                    :,
                ]
        else:
            if len(image.shape) == 4:
                image = image[
                    :,
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                ]
            else:
                image = image[
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                ]
    elif pad_to_aspect_ratio:
        shape = image.shape
        batch_size = image.shape[0]
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
            if len(image.shape) == 4:
                padded_img = (
                    mx.ones(
                        (
                            batch_size,
                            pad_height + height,
                            pad_width + width,
                            channels,
                        ),
                        dtype=image.dtype,
                    )
                    * fill_value
                )
                padded_img[
                    :,
                    img_box_hstart : img_box_hstart + height,
                    img_box_wstart : img_box_wstart + width,
                    :,
                ] = image
            else:
                padded_img = (
                    mx.ones(
                        (pad_height + height, pad_width + width, channels),
                        dtype=image.dtype,
                    )
                    * fill_value
                )
                padded_img[
                    img_box_hstart : img_box_hstart + height,
                    img_box_wstart : img_box_wstart + width,
                    :,
                ] = image
        else:
            if len(image.shape) == 4:
                padded_img = (
                    mx.ones(
                        (
                            batch_size,
                            channels,
                            pad_height + height,
                            pad_width + width,
                        ),
                        dtype=image.dtype,
                    )
                    * fill_value
                )
                padded_img[
                    :,
                    :,
                    img_box_hstart : img_box_hstart + height,
                    img_box_wstart : img_box_wstart + width,
                ] = image
            else:
                padded_img = (
                    mx.ones(
                        (channels, pad_height + height, pad_width + width),
                        dtype=image.dtype,
                    )
                    * fill_value
                )
                padded_img[
                    :,
                    img_box_hstart : img_box_hstart + height,
                    img_box_wstart : img_box_wstart + width,
                ] = image
        image = padded_img

    # Change to channels_last
    if data_format == "channels_first":
        image = (
            image.transpose(0, 2, 3, 1)
            if image.ndim == 4
            else image.transpose(1, 2, 0)
        )

    *_, H, W, C = image.shape
    transform = mx.array([H / size[0], 0, 0, 0, W / size[1], 0, 0, 0])
    result = _affine_transform(
        src=image,
        transform=transform,
        target_size=size,
        interpolation_function=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
        index_fixer=_INDEX_FIXERS["constant"],
        fill_value=0,
        check_validity=False,
    )

    # Change back to channels_first
    if data_format == "channels_first":
        result = (
            result.transpose(0, 3, 1, 2)
            if image.ndim == 4
            else result.transpose(2, 0, 1)
        )

    return result
