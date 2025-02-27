import functools
import itertools
import operator

import torch
import torch.nn.functional as F

from keras.src import backend
from keras.src.backend.torch.core import convert_to_tensor

RESIZE_INTERPOLATIONS = {
    "bilinear": "bilinear",
    "nearest": "nearest-exact",
    "bicubic": "bicubic",
}

UNSUPPORTED_INTERPOLATIONS = (
    "lanczos3",
    "lanczos5",
)


def rgb_to_grayscale(images, data_format=None):
    images = convert_to_tensor(images)
    data_format = backend.standardize_data_format(data_format)
    if images.ndim not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    channel_axis = -3 if data_format == "channels_first" else -1
    if images.shape[channel_axis] not in (1, 3):
        raise ValueError(
            "Invalid channel size: expected 3 (RGB) or 1 (Grayscale). "
            f"Received input with shape: images.shape={images.shape}"
        )

    # This implementation is based on
    # https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py
    if images.shape[channel_axis] == 3:
        r, g, b = images.unbind(dim=channel_axis)
        images = (0.2989 * r + 0.587 * g + 0.114 * b).to(images.dtype)
        images = images.unsqueeze(dim=channel_axis)
    else:
        images = images.clone()
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


def _cast_squeeze_in(image, req_dtypes):
    need_squeeze = False
    # make image NCHW
    if image.ndim < 4:
        image = image.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = image.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        image = image.to(req_dtype)
    return image, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(image, need_cast, need_squeeze, out_dtype):
    if need_squeeze:
        image = image.squeeze(dim=0)

    if need_cast:
        if out_dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            # it is better to round before cast
            image = torch.round(image)
        image = image.to(out_dtype)
    return image


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
    images, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(
        images, [torch.float32, torch.float64]
    )
    if data_format == "channels_last":
        images = images.permute((0, 3, 1, 2))

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
        images = images[
            :,
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

        batch_size = images.shape[0]
        channels = images.shape[1]
        if img_box_hstart > 0:
            padded_img = torch.cat(
                [
                    torch.ones(
                        (batch_size, channels, img_box_hstart, width),
                        dtype=images.dtype,
                        device=images.device,
                    )
                    * fill_value,
                    images,
                    torch.ones(
                        (batch_size, channels, img_box_hstart, width),
                        dtype=images.dtype,
                        device=images.device,
                    )
                    * fill_value,
                ],
                axis=2,
            )
        else:
            padded_img = images
        if img_box_wstart > 0:
            padded_img = torch.cat(
                [
                    torch.ones(
                        (batch_size, channels, height, img_box_wstart),
                        dtype=images.dtype,
                        device=images.device,
                    ),
                    padded_img,
                    torch.ones(
                        (batch_size, channels, height, img_box_wstart),
                        dtype=images.dtype,
                        device=images.device,
                    )
                    * fill_value,
                ],
                axis=3,
            )
        images = padded_img

    # This implementation is based on
    # https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py
    if antialias and interpolation not in ("bilinear", "bicubic"):
        # We manually set it to False to avoid an error downstream in
        # interpolate(). This behaviour is documented: the parameter is
        # irrelevant for modes that are not bilinear or bicubic. We used to
        # raise an error here, but now we don't use True as the default.
        antialias = False
    # Define align_corners to avoid warnings
    align_corners = False if interpolation in ("bilinear", "bicubic") else None
    resized = F.interpolate(
        images,
        size=size,
        mode=RESIZE_INTERPOLATIONS[interpolation],
        align_corners=align_corners,
        antialias=antialias,
    )
    if interpolation == "bicubic" and out_dtype == torch.uint8:
        resized = resized.clamp(min=0, max=255)
    if data_format == "channels_last":
        resized = resized.permute((0, 2, 3, 1))
    resized = _cast_squeeze_out(
        resized,
        need_cast=need_cast,
        need_squeeze=need_squeeze,
        out_dtype=out_dtype,
    )
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


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)

    images = convert_to_tensor(images)
    start_points = torch.tensor(start_points, dtype=torch.float32)
    end_points = torch.tensor(end_points, dtype=torch.float32)

    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS.keys():
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )

    if images.ndim not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    if start_points.shape[-2:] != (4, 2) or start_points.dim() not in (2, 3):
        raise ValueError(
            "Invalid start_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {start_points.shape}"
        )
    if end_points.shape[-2:] != (4, 2) or end_points.dim() not in (2, 3):
        raise ValueError(
            "Invalid end_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {end_points.shape}"
        )
    if start_points.shape != end_points.shape:
        raise ValueError(
            "start_points and end_points must have the same shape."
            f" Received start_points.shape={start_points.shape}, "
            f"end_points.shape={end_points.shape}"
        )

    need_squeeze = False
    if images.ndim == 3:
        images = images.unsqueeze(dim=0)
        need_squeeze = True

    if start_points.ndim == 2:
        start_points = start_points.unsqueeze(dim=0)
    if end_points.ndim == 2:
        end_points = end_points.unsqueeze(dim=0)

    if data_format == "channels_first":
        images = images.permute((0, 2, 3, 1))

    batch_size, height, width, channels = images.shape

    transforms = compute_homography_matrix(start_points, end_points)

    if transforms.dim() == 1:
        transforms = transforms.unsqueeze(0)
    if transforms.shape[0] == 1 and batch_size > 1:
        transforms = transforms.repeat(batch_size, 1)

    grid_x, grid_y = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=images.device),
        torch.arange(height, dtype=torch.float32, device=images.device),
        indexing="xy",
    )

    output = torch.empty(
        [batch_size, height, width, channels], device=images.device
    )

    for i in range(batch_size):
        a0, a1, a2, a3, a4, a5, a6, a7 = transforms[i]
        denom = a6 * grid_x + a7 * grid_y + 1.0
        x_in = (a0 * grid_x + a1 * grid_y + a2) / denom
        y_in = (a3 * grid_x + a4 * grid_y + a5) / denom

        coords = torch.stack([y_in.flatten(), x_in.flatten()], dim=0)
        mapped_channels = []
        for channel in range(channels):
            channel_img = images[i, :, :, channel]
            mapped_channel = map_coordinates(
                channel_img,
                coords,
                order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                fill_mode="constant",
                fill_value=fill_value,
            )
            mapped_channels.append(mapped_channel.reshape(height, width))
        output[i] = torch.stack(mapped_channels, dim=-1)

    if data_format == "channels_first":
        output = output.permute((0, 3, 1, 2))
    if need_squeeze:
        output = output.squeeze(dim=0)

    return output


def compute_homography_matrix(start_points, end_points):
    start_points = convert_to_tensor(start_points, dtype=torch.float32)
    end_points = convert_to_tensor(end_points, dtype=torch.float32)

    start_x1, start_y1 = start_points[:, 0, 0], start_points[:, 0, 1]
    start_x2, start_y2 = start_points[:, 1, 0], start_points[:, 1, 1]
    start_x3, start_y3 = start_points[:, 2, 0], start_points[:, 2, 1]
    start_x4, start_y4 = start_points[:, 3, 0], start_points[:, 3, 1]

    end_x1, end_y1 = end_points[:, 0, 0], end_points[:, 0, 1]
    end_x2, end_y2 = end_points[:, 1, 0], end_points[:, 1, 1]
    end_x3, end_y3 = end_points[:, 2, 0], end_points[:, 2, 1]
    end_x4, end_y4 = end_points[:, 3, 0], end_points[:, 3, 1]

    coefficient_matrix = torch.stack(
        [
            torch.stack(
                [
                    end_x1,
                    end_y1,
                    torch.ones_like(end_x1),
                    torch.zeros_like(end_x1),
                    torch.zeros_like(end_x1),
                    torch.zeros_like(end_x1),
                    -start_x1 * end_x1,
                    -start_x1 * end_y1,
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    torch.zeros_like(end_x1),
                    torch.zeros_like(end_x1),
                    torch.zeros_like(end_x1),
                    end_x1,
                    end_y1,
                    torch.ones_like(end_x1),
                    -start_y1 * end_x1,
                    -start_y1 * end_y1,
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    end_x2,
                    end_y2,
                    torch.ones_like(end_x2),
                    torch.zeros_like(end_x2),
                    torch.zeros_like(end_x2),
                    torch.zeros_like(end_x2),
                    -start_x2 * end_x2,
                    -start_x2 * end_y2,
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    torch.zeros_like(end_x2),
                    torch.zeros_like(end_x2),
                    torch.zeros_like(end_x2),
                    end_x2,
                    end_y2,
                    torch.ones_like(end_x2),
                    -start_y2 * end_x2,
                    -start_y2 * end_y2,
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    end_x3,
                    end_y3,
                    torch.ones_like(end_x3),
                    torch.zeros_like(end_x3),
                    torch.zeros_like(end_x3),
                    torch.zeros_like(end_x3),
                    -start_x3 * end_x3,
                    -start_x3 * end_y3,
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    torch.zeros_like(end_x3),
                    torch.zeros_like(end_x3),
                    torch.zeros_like(end_x3),
                    end_x3,
                    end_y3,
                    torch.ones_like(end_x3),
                    -start_y3 * end_x3,
                    -start_y3 * end_y3,
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    end_x4,
                    end_y4,
                    torch.ones_like(end_x4),
                    torch.zeros_like(end_x4),
                    torch.zeros_like(end_x4),
                    torch.zeros_like(end_x4),
                    -start_x4 * end_x4,
                    -start_x4 * end_y4,
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    torch.zeros_like(end_x4),
                    torch.zeros_like(end_x4),
                    torch.zeros_like(end_x4),
                    end_x4,
                    end_y4,
                    torch.ones_like(end_x4),
                    -start_y4 * end_x4,
                    -start_y4 * end_y4,
                ],
                dim=-1,
            ),
        ],
        dim=1,
    )

    target_vector = torch.stack(
        [
            start_x1,
            start_y1,
            start_x2,
            start_y2,
            start_x3,
            start_y3,
            start_x4,
            start_y4,
        ],
        dim=-1,
    ).unsqueeze(-1)

    homography_matrix = torch.linalg.solve(coefficient_matrix, target_vector)
    homography_matrix = homography_matrix.reshape(-1, 8)

    return homography_matrix


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


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    def _create_gaussian_kernel(kernel_size, sigma, dtype):
        def _get_gaussian_kernel1d(size, sigma):
            x = (
                torch.arange(size, dtype=dtype, device=sigma.device)
                - (size - 1) / 2
            )
            kernel1d = torch.exp(-0.5 * (x / sigma) ** 2)
            return kernel1d / torch.sum(kernel1d)

        def _get_gaussian_kernel2d(size, sigma):
            size = torch.tensor(size, dtype=dtype)
            kernel1d_x = _get_gaussian_kernel1d(size[0], sigma[0])
            kernel1d_y = _get_gaussian_kernel1d(size[1], sigma[1])
            return torch.outer(kernel1d_y, kernel1d_x)

        kernel = _get_gaussian_kernel2d(kernel_size, sigma)

        kernel = kernel.view(1, 1, kernel_size[0], kernel_size[1])
        return kernel

    images = convert_to_tensor(images)
    kernel_size = convert_to_tensor(kernel_size)
    sigma = convert_to_tensor(sigma)
    dtype = images.dtype

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    need_squeeze = False
    if images.ndim == 3:
        images = images.unsqueeze(dim=0)
        need_squeeze = True

    if data_format == "channels_last":
        images = images.permute(0, 3, 1, 2)

    num_channels = images.shape[1]
    kernel = _create_gaussian_kernel(kernel_size, sigma, dtype)

    kernel = kernel.expand(num_channels, 1, kernel_size[0], kernel_size[1])

    print(kernel_size[0] // 2)

    blurred_images = torch.nn.functional.conv2d(
        images,
        kernel,
        stride=1,
        padding=int(kernel_size[0] // 2),
        groups=num_channels,
    )

    if data_format == "channels_last":
        blurred_images = blurred_images.permute(0, 2, 3, 1)

    if need_squeeze:
        blurred_images = blurred_images.squeeze(dim=0)

    return blurred_images
