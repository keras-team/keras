import functools
import itertools
import operator

import torch

from keras.src.backend.torch.core import convert_to_tensor

RESIZE_INTERPOLATIONS = {}  # populated after torchvision import

UNSUPPORTED_INTERPOLATIONS = (
    "lanczos3",
    "lanczos5",
)


def rgb_to_grayscale(image, data_format="channel_last"):
    try:
        import torchvision
    except:
        raise ImportError(
            "The torchvision package is necessary to use "
            "`rgb_to_grayscale` with the torch backend. "
            "Please install torchvision. "
        )
    image = convert_to_tensor(image)
    if data_format == "channels_last":
        if image.ndim == 4:
            image = image.permute((0, 3, 1, 2))
        elif image.ndim == 3:
            image = image.permute((2, 0, 1))
        else:
            raise ValueError(
                "Invalid input rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"image.shape={image.shape}"
            )
    grayscale_image = torchvision.transforms.functional.rgb_to_grayscale(
        img=image,
    )
    if data_format == "channels_last":
        if len(image.shape) == 4:
            grayscale_image = grayscale_image.permute((0, 2, 3, 1))
        elif len(image.shape) == 3:
            grayscale_image = grayscale_image.permute((1, 2, 0))
    return grayscale_image


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
    try:
        import torchvision
        from torchvision.transforms import InterpolationMode as im

        RESIZE_INTERPOLATIONS.update(
            {
                "bilinear": im.BILINEAR,
                "nearest": im.NEAREST_EXACT,
                "bicubic": im.BICUBIC,
            }
        )
    except:
        raise ImportError(
            "The torchvision package is necessary to use `resize` with the "
            "torch backend. Please install torchvision."
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
    image = convert_to_tensor(image)
    if data_format == "channels_last":
        if image.ndim == 4:
            image = image.permute((0, 3, 1, 2))
        elif image.ndim == 3:
            image = image.permute((2, 0, 1))
        else:
            raise ValueError(
                "Invalid input rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"image.shape={image.shape}"
            )

    if crop_to_aspect_ratio:
        shape = image.shape
        height, width = shape[-2], shape[-1]
        target_height, target_width = size
        crop_height = int(float(width * target_height) / target_width)
        crop_height = min(height, crop_height)
        crop_width = int(float(height * target_width) / target_height)
        crop_width = min(width, crop_width)
        crop_box_hstart = int(float(height - crop_height) / 2)
        crop_box_wstart = int(float(width - crop_width) / 2)
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
        height, width = shape[-2], shape[-1]
        target_height, target_width = size
        pad_height = int(float(width * target_height) / target_width)
        pad_height = max(height, pad_height)
        pad_width = int(float(height * target_width) / target_height)
        pad_width = max(width, pad_width)
        img_box_hstart = int(float(pad_height - height) / 2)
        img_box_wstart = int(float(pad_width - width) / 2)
        if len(image.shape) == 4:
            batch_size = image.shape[0]
            channels = image.shape[1]
            padded_img = (
                torch.ones(
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
            channels = image.shape[0]
            padded_img = (
                torch.ones(
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

    resized = torchvision.transforms.functional.resize(
        img=image,
        size=size,
        interpolation=RESIZE_INTERPOLATIONS[interpolation],
        antialias=antialias,
    )
    if data_format == "channels_last":
        if len(image.shape) == 4:
            resized = resized.permute((0, 2, 3, 1))
        elif len(image.shape) == 3:
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

    # unbatched case
    need_squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(dim=0)
        need_squeeze = True
    if transform.ndim == 1:
        transform = transform.unsqueeze(dim=0)

    if data_format == "channels_first":
        image = image.permute((0, 2, 3, 1))

    batch_size = image.shape[0]

    # get indices
    meshgrid = torch.meshgrid(
        *[
            torch.arange(size, dtype=transform.dtype, device=transform.device)
            for size in image.shape[1:]
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
                image[i],
                coordinates[i],
                order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                fill_mode=fill_mode,
                fill_value=fill_value,
            )
            for i in range(len(image))
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
