import functools
import itertools
import operator

import torch

from keras.src import backend
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.utils.module_utils import torchvision

RESIZE_INTERPOLATIONS = {}  # populated after torchvision import

UNSUPPORTED_INTERPOLATIONS = (
    "lanczos3",
    "lanczos5",
)


def rgb_to_grayscale(images, data_format=None):
    images = convert_to_tensor(images)
    data_format = backend.standardize_data_format(data_format)
    if data_format == "channels_last":
        if images.ndim == 4:
            images = images.permute((0, 3, 1, 2))
        elif images.ndim == 3:
            images = images.permute((2, 0, 1))
        else:
            raise ValueError(
                "Invalid images rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"images.shape={images.shape}"
            )
    images = torchvision.transforms.functional.rgb_to_grayscale(img=images)
    if data_format == "channels_last":
        if len(images.shape) == 4:
            images = images.permute((0, 2, 3, 1))
        elif len(images.shape) == 3:
            images = images.permute((1, 2, 0))
    return images


def rgb_to_hsv(images, data_format=None):
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
    eps = torch.finfo(dtype).eps
    images = torch.where(torch.abs(images) < eps, 0.0, images)
    red, green, blue = torch.split(images, [1, 1, 1], channels_axis)
    red = torch.squeeze(red, channels_axis)
    green = torch.squeeze(green, channels_axis)
    blue = torch.squeeze(blue, channels_axis)

    def rgb_planes_to_hsv_planes(r, g, b):
        value = torch.maximum(torch.maximum(r, g), b)
        minimum = torch.minimum(torch.minimum(r, g), b)
        range_ = value - minimum

        safe_value = torch.where(value > 0, value, 1.0)
        safe_range = torch.where(range_ > 0, range_, 1.0)

        saturation = torch.where(value > 0, range_ / safe_value, 0.0)
        norm = 1.0 / (6.0 * safe_range)

        hue = torch.where(
            value == g,
            norm * (b - r) + 2.0 / 6.0,
            norm * (r - g) + 4.0 / 6.0,
        )
        hue = torch.where(value == r, norm * (g - b), hue)
        hue = torch.where(range_ > 0, hue, 0.0) + (hue < 0.0).to(hue.dtype)
        return hue, saturation, value

    images = torch.stack(
        rgb_planes_to_hsv_planes(red, green, blue), axis=channels_axis
    )
    return images


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
    hue, saturation, value = torch.split(images, [1, 1, 1], channels_axis)
    hue = torch.squeeze(hue, channels_axis)
    saturation = torch.squeeze(saturation, channels_axis)
    value = torch.squeeze(value, channels_axis)

    def hsv_planes_to_rgb_planes(hue, saturation, value):
        dh = torch.remainder(hue, 1.0) * 6.0
        dr = torch.clip(torch.abs(dh - 3.0) - 1.0, 0.0, 1.0)
        dg = torch.clip(2.0 - torch.abs(dh - 2.0), 0.0, 1.0)
        db = torch.clip(2.0 - torch.abs(dh - 4.0), 0.0, 1.0)
        one_minus_s = 1.0 - saturation

        red = value * (one_minus_s + saturation * dr)
        green = value * (one_minus_s + saturation * dg)
        blue = value * (one_minus_s + saturation * db)
        return red, green, blue

    images = torch.stack(
        hsv_planes_to_rgb_planes(hue, saturation, value), axis=channels_axis
    )
    return images


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
    RESIZE_INTERPOLATIONS.update(
        {
            "bilinear": torchvision.transforms.InterpolationMode.BILINEAR,
            "nearest": torchvision.transforms.InterpolationMode.NEAREST_EXACT,
            "bicubic": torchvision.transforms.InterpolationMode.BICUBIC,
        }
    )
    if interpolation in UNSUPPORTED_INTERPOLATIONS:
        raise ValueError(
            "Resizing with Lanczos interpolation is "
            "not supported by the PyTorch backend. "
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
    images = convert_to_tensor(images)
    if images.ndim not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if data_format == "channels_last":
        if images.ndim == 4:
            images = images.permute((0, 3, 1, 2))
        else:
            images = images.permute((2, 0, 1))

    if crop_to_aspect_ratio:
        shape = images.shape
        height, width = shape[-2], shape[-1]
        target_height, target_width = size
        crop_height = int(float(width * target_height) / target_width)
        crop_height = max(min(height, crop_height), 1)
        crop_width = int(float(height * target_width) / target_height)
        crop_width = max(min(width, crop_width), 1)
        crop_box_hstart = int(float(height - crop_height) / 2)
        crop_box_wstart = int(float(width - crop_width) / 2)
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
        height, width = shape[-2], shape[-1]
        target_height, target_width = size
        pad_height = int(float(width * target_height) / target_width)
        pad_height = max(height, pad_height)
        pad_width = int(float(height * target_width) / target_height)
        pad_width = max(width, pad_width)
        img_box_hstart = int(float(pad_height - height) / 2)
        img_box_wstart = int(float(pad_width - width) / 2)
        if len(images.shape) == 4:
            batch_size = images.shape[0]
            channels = images.shape[1]
            padded_img = (
                torch.ones(
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
            channels = images.shape[0]
            padded_img = (
                torch.ones(
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

    resized = torchvision.transforms.functional.resize(
        img=images,
        size=size,
        interpolation=RESIZE_INTERPOLATIONS[interpolation],
        antialias=antialias,
    )
    if data_format == "channels_last":
        if len(images.shape) == 4:
            resized = resized.permute((0, 2, 3, 1))
        elif len(images.shape) == 3:
            resized = resized.permute((1, 2, 0))
    return resized


AFFINE_TRANSFORM_INTERPOLATIONS = {
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

    images = convert_to_tensor(images)
    transform = convert_to_tensor(transform)

    if images.ndim not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if transform.ndim not in (1, 2):
        raise ValueError(
            "Invalid transform rank: expected rank 1 (single transform) "
            "or rank 2 (batch of transforms). Received input with shape: "
            f"transform.shape={transform.shape}"
        )

    # unbatched case
    need_squeeze = False
    if images.ndim == 3:
        images = images.unsqueeze(dim=0)
        need_squeeze = True
    if transform.ndim == 1:
        transform = transform.unsqueeze(dim=0)

    if data_format == "channels_first":
        images = images.permute((0, 2, 3, 1))

    batch_size = images.shape[0]

    # get indices
    meshgrid = torch.meshgrid(
        *[
            torch.arange(size, dtype=transform.dtype, device=transform.device)
            for size in images.shape[1:]
        ],
        indexing="ij",
    )
    indices = torch.concatenate(
        [torch.unsqueeze(x, dim=-1) for x in meshgrid], dim=-1
    )
    indices = torch.tile(indices, (batch_size, 1, 1, 1, 1))

    # swap the values
    a0 = transform[:, 0].clone()
    a2 = transform[:, 2].clone()
    b1 = transform[:, 4].clone()
    b2 = transform[:, 5].clone()
    transform[:, 0] = b1
    transform[:, 2] = b2
    transform[:, 4] = a0
    transform[:, 5] = a2

    # deal with transform
    transform = torch.nn.functional.pad(
        transform, pad=[0, 1, 0, 0], mode="constant", value=1
    )
    transform = torch.reshape(transform, (batch_size, 3, 3))
    offset = transform[:, 0:2, 2].clone()
    offset = torch.nn.functional.pad(offset, pad=[0, 1, 0, 0])
    transform[:, 0:2, 2] = 0

    # transform the indices
    coordinates = torch.einsum("Bhwij, Bjk -> Bhwik", indices, transform)
    coordinates = torch.moveaxis(coordinates, source=-1, destination=1)
    coordinates += torch.reshape(a=offset, shape=(*offset.shape, 1, 1, 1))

    # Note: torch.stack is faster than torch.vmap when the batch size is small.
    affined = torch.stack(
        [
            map_coordinates(
                images[i],
                coordinates[i],
                order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                fill_mode=fill_mode,
                fill_value=fill_value,
            )
            for i in range(len(images))
        ],
    )

    if data_format == "channels_first":
        affined = affined.permute((0, 3, 1, 2))
    if need_squeeze:
        affined = affined.squeeze(dim=0)
    return affined


def _mirror_index_fixer(index, size):
    s = size - 1  # Half-wavelength of triangular wave
    # Scaled, integer-valued version of the triangular wave |x - round(x)|
    return torch.abs((index + s) % (2 * s) - s)


def _reflect_index_fixer(index, size):
    return torch.floor_divide(
        _mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2
    )


_INDEX_FIXERS = {
    # we need to take care of out-of-bound indices in torch
    "constant": lambda index, size: torch.clip(index, 0, size - 1),
    "nearest": lambda index, size: torch.clip(index, 0, size - 1),
    "wrap": lambda index, size: index % size,
    "mirror": _mirror_index_fixer,
    "reflect": _reflect_index_fixer,
}


def _is_integer(a):
    if not torch.is_floating_point(a) and not torch.is_complex(a):
        return True
    return False


def _nearest_indices_and_weights(coordinate):
    coordinate = (
        coordinate if _is_integer(coordinate) else torch.round(coordinate)
    )
    index = coordinate.to(torch.int32)
    return [(index, 1)]


def _linear_indices_and_weights(coordinate):
    lower = torch.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.to(torch.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0.0
):
    input_arr = convert_to_tensor(inputs)
    coordinate_arrs = [convert_to_tensor(c) for c in coordinates]

    if len(coordinate_arrs) != len(input_arr.shape):
        raise ValueError(
            "First dim of `coordinates` must be the same as the rank of "
            "`inputs`. "
            f"Received inputs with shape: {input_arr.shape} and coordinate "
            f"leading dim of {len(coordinate_arrs)}"
        )
    if len(coordinate_arrs[0].shape) < 1:
        dim = len(coordinate_arrs)
        shape = (dim,) + coordinate_arrs[0].shape
        raise ValueError(
            "Invalid coordinates rank: expected at least rank 2."
            f" Received input with shape: {shape}"
        )

    # skip tensor creation as possible
    if isinstance(fill_value, (int, float)) and _is_integer(input_arr):
        fill_value = int(fill_value)

    if len(coordinates) != len(input_arr.shape):
        raise ValueError(
            "coordinates must be a sequence of length inputs.shape, but "
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

    if fill_mode == "constant":

        def is_valid(index, size):
            return (0 <= index) & (index < size)

    else:

        def is_valid(index, size):
            return True

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinate_arrs, input_arr.shape):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = zip(*items)
        if all(valid is True for valid in validities):
            # fast path
            contribution = input_arr[indices]
        else:
            all_valid = functools.reduce(operator.and_, validities)
            contribution = torch.where(
                all_valid, input_arr[indices], fill_value
            )
        outputs.append(functools.reduce(operator.mul, weights) * contribution)
    result = functools.reduce(operator.add, outputs)
    if _is_integer(input_arr):
        result = result if _is_integer(result) else torch.round(result)
    return result.to(input_arr.dtype)
