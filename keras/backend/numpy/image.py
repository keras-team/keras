import jax
import numpy as np

from keras.backend.numpy.core import convert_to_tensor
from keras.utils.module_utils import scipy

RESIZE_INTERPOLATIONS = (
    "bilinear",
    "nearest",
    "lanczos3",
    "lanczos5",
    "bicubic",
)


def resize(
    image,
    size,
    interpolation="bilinear",
    antialias=False,
    data_format="channels_last",
):
    if interpolation not in RESIZE_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{RESIZE_INTERPOLATIONS}. Received: interpolation={interpolation}"
        )
    if not len(size) == 2:
        raise ValueError(
            "Argument `size` must be a tuple of two elements "
            f"(height, width). Received: size={size}"
        )
    size = tuple(size)
    if len(image.shape) == 4:
        if data_format == "channels_last":
            size = (image.shape[0],) + size + (image.shape[-1],)
        else:
            size = (image.shape[0], image.shape[1]) + size
    elif len(image.shape) == 3:
        if data_format == "channels_last":
            size = size + (image.shape[-1],)
        else:
            size = (image.shape[0],) + size
    else:
        raise ValueError(
            "Invalid input rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"image.shape={image.shape}"
        )
    return np.array(
        jax.image.resize(image, size, method=interpolation, antialias=antialias)
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

    transform = convert_to_tensor(transform)

    if len(image.shape) not in (3, 4):
        raise ValueError(
            "Invalid image rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"image.shape={image.shape}"
        )
    if len(transform.shape) not in (1, 2):
        raise ValueError(
            "Invalid transform rank: expected rank 1 (single transform) "
            "or rank 2 (batch of transforms). Received input with shape: "
            f"transform.shape={transform.shape}"
        )

    # scipy.ndimage.map_coordinates lacks support for half precision.
    input_dtype = image.dtype
    if input_dtype == "float16":
        image = image.astype("float32")

    # unbatched case
    need_squeeze = False
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
        need_squeeze = True
    if len(transform.shape) == 1:
        transform = np.expand_dims(transform, axis=0)

    if data_format == "channels_first":
        image = np.transpose(image, (0, 2, 3, 1))

    batch_size = image.shape[0]

    # get indices
    meshgrid = np.meshgrid(
        *[np.arange(size) for size in image.shape[1:]], indexing="ij"
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
                image[i],
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
    input, coordinates, order, fill_mode="constant", fill_value=0.0
):
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
        for c, size in zip(coordinates, input.shape)
    ]
    shifted_coords = [c + p[0] for p, c in zip(padding, coordinates)]
    pad_mode = {
        "nearest": "edge",
        "mirror": "reflect",
        "reflect": "symmetric",
    }.get(fill_mode, fill_mode)
    if fill_mode == "constant":
        padded = np.pad(
            input, padding, mode=pad_mode, constant_values=fill_value
        )
    else:
        padded = np.pad(input, padding, mode=pad_mode)
    result = scipy.ndimage.map_coordinates(
        padded, shifted_coords, order=order, mode=fill_mode, cval=fill_value
    )
    return result
