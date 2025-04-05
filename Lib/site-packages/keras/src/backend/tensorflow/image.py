import functools
import itertools
import operator

import tensorflow as tf

from keras.src import backend
from keras.src.backend.tensorflow.core import convert_to_tensor

RESIZE_INTERPOLATIONS = (
    "bilinear",
    "nearest",
    "lanczos3",
    "lanczos5",
    "bicubic",
    "area",
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
    images = tf.cast(images, compute_dtype)

    # Ref: tf.image.rgb_to_grayscale
    rgb_weights = convert_to_tensor(
        [0.2989, 0.5870, 0.1140], dtype=images.dtype
    )
    images = tf.tensordot(images, rgb_weights, axes=(channels_axis, -1))
    images = tf.expand_dims(images, axis=channels_axis)
    return tf.cast(images, original_dtype)


def rgb_to_hsv(images, data_format=None):
    images = convert_to_tensor(images)
    dtype = images.dtype
    data_format = backend.standardize_data_format(data_format)
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
    if data_format == "channels_first":
        if len(images.shape) == 4:
            images = tf.transpose(images, (0, 2, 3, 1))
        else:
            images = tf.transpose(images, (1, 2, 0))
    images = tf.image.rgb_to_hsv(images)
    if data_format == "channels_first":
        if len(images.shape) == 4:
            images = tf.transpose(images, (0, 3, 1, 2))
        elif len(images.shape) == 3:
            images = tf.transpose(images, (2, 0, 1))
    return images


def hsv_to_rgb(images, data_format=None):
    images = convert_to_tensor(images)
    dtype = images.dtype
    data_format = backend.standardize_data_format(data_format)
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
    if data_format == "channels_first":
        if len(images.shape) == 4:
            images = tf.transpose(images, (0, 2, 3, 1))
        else:
            images = tf.transpose(images, (1, 2, 0))
    images = tf.image.hsv_to_rgb(images)
    if data_format == "channels_first":
        if len(images.shape) == 4:
            images = tf.transpose(images, (0, 3, 1, 2))
        elif len(images.shape) == 3:
            images = tf.transpose(images, (2, 0, 1))
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
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if data_format == "channels_first":
        if len(images.shape) == 4:
            images = tf.transpose(images, (0, 2, 3, 1))
        else:
            images = tf.transpose(images, (1, 2, 0))

    if crop_to_aspect_ratio:
        shape = tf.shape(images)
        height, width = shape[-3], shape[-2]
        target_height, target_width = size
        crop_height = tf.cast(
            tf.cast(width * target_height, "float32") / target_width,
            "int32",
        )
        crop_height = tf.maximum(tf.minimum(height, crop_height), 1)
        crop_height = tf.cast(crop_height, "int32")
        crop_width = tf.cast(
            tf.cast(height * target_width, "float32") / target_height,
            "int32",
        )
        crop_width = tf.maximum(tf.minimum(width, crop_width), 1)
        crop_width = tf.cast(crop_width, "int32")

        crop_box_hstart = tf.cast(
            tf.cast(height - crop_height, "float32") / 2, "int32"
        )
        crop_box_wstart = tf.cast(
            tf.cast(width - crop_width, "float32") / 2, "int32"
        )
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
    elif pad_to_aspect_ratio:
        shape = tf.shape(images)
        height, width = shape[-3], shape[-2]
        target_height, target_width = size
        pad_height = tf.cast(
            tf.cast(width * target_height, "float32") / target_width,
            "int32",
        )
        pad_height = tf.maximum(height, pad_height)
        pad_height = tf.cast(pad_height, "int32")
        pad_width = tf.cast(
            tf.cast(height * target_width, "float32") / target_height,
            "int32",
        )
        pad_width = tf.maximum(width, pad_width)
        pad_width = tf.cast(pad_width, "int32")

        img_box_hstart = tf.cast(
            tf.cast(pad_height - height, "float32") / 2, "int32"
        )
        img_box_wstart = tf.cast(
            tf.cast(pad_width - width, "float32") / 2, "int32"
        )
        if len(images.shape) == 4:
            batch_size = tf.shape(images)[0]
            channels = tf.shape(images)[3]
            padded_img = tf.cond(
                img_box_hstart > 0,
                lambda: tf.concat(
                    [
                        tf.ones(
                            (batch_size, img_box_hstart, width, channels),
                            dtype=images.dtype,
                        )
                        * fill_value,
                        images,
                        tf.ones(
                            (batch_size, img_box_hstart, width, channels),
                            dtype=images.dtype,
                        )
                        * fill_value,
                    ],
                    axis=1,
                ),
                lambda: images,
            )
            padded_img = tf.cond(
                img_box_wstart > 0,
                lambda: tf.concat(
                    [
                        tf.ones(
                            (batch_size, height, img_box_wstart, channels),
                            dtype=images.dtype,
                        )
                        * fill_value,
                        padded_img,
                        tf.ones(
                            (batch_size, height, img_box_wstart, channels),
                            dtype=images.dtype,
                        )
                        * fill_value,
                    ],
                    axis=2,
                ),
                lambda: padded_img,
            )
        else:
            channels = tf.shape(images)[2]
            padded_img = tf.cond(
                img_box_hstart > 0,
                lambda: tf.concat(
                    [
                        tf.ones(
                            (img_box_hstart, width, channels),
                            dtype=images.dtype,
                        )
                        * fill_value,
                        images,
                        tf.ones(
                            (img_box_hstart, width, channels),
                            dtype=images.dtype,
                        )
                        * fill_value,
                    ],
                    axis=0,
                ),
                lambda: images,
            )
            padded_img = tf.cond(
                img_box_wstart > 0,
                lambda: tf.concat(
                    [
                        tf.ones(
                            (height, img_box_wstart, channels),
                            dtype=images.dtype,
                        )
                        * fill_value,
                        padded_img,
                        tf.ones(
                            (height, img_box_wstart, channels),
                            dtype=images.dtype,
                        )
                        * fill_value,
                    ],
                    axis=1,
                ),
                lambda: padded_img,
            )
        images = padded_img

    resized = tf.image.resize(
        images, size, method=interpolation, antialias=antialias
    )
    if data_format == "channels_first":
        if len(images.shape) == 4:
            resized = tf.transpose(resized, (0, 3, 1, 2))
        elif len(images.shape) == 3:
            resized = tf.transpose(resized, (2, 0, 1))
    return resized


AFFINE_TRANSFORM_INTERPOLATIONS = (
    "nearest",
    "bilinear",
)
AFFINE_TRANSFORM_FILL_MODES = (
    "constant",
    "nearest",
    "wrap",
    # "mirror", not supported by TF
    "reflect",
)


def affine_transform(
    images,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{AFFINE_TRANSFORM_INTERPOLATIONS}. Received: "
            f"interpolation={interpolation}"
        )
    if fill_mode not in AFFINE_TRANSFORM_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{AFFINE_TRANSFORM_FILL_MODES}. Received: fill_mode={fill_mode}"
        )
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
    # unbatched case
    need_squeeze = False
    if len(images.shape) == 3:
        images = tf.expand_dims(images, axis=0)
        need_squeeze = True
    if len(transform.shape) == 1:
        transform = tf.expand_dims(transform, axis=0)

    if data_format == "channels_first":
        images = tf.transpose(images, (0, 2, 3, 1))

    affined = tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=tf.cast(transform, dtype=tf.float32),
        output_shape=tf.shape(images)[1:-1],
        fill_value=fill_value,
        interpolation=interpolation.upper(),
        fill_mode=fill_mode.upper(),
    )
    affined = tf.ensure_shape(affined, images.shape)

    if data_format == "channels_first":
        affined = tf.transpose(affined, (0, 3, 1, 2))
    if need_squeeze:
        affined = tf.squeeze(affined, axis=0)
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
    start_points = convert_to_tensor(start_points, dtype=tf.float32)
    end_points = convert_to_tensor(end_points, dtype=tf.float32)

    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{AFFINE_TRANSFORM_INTERPOLATIONS}. Received: "
            f"interpolation={interpolation}"
        )
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    if start_points.shape.rank not in (2, 3) or start_points.shape[-2:] != (
        4,
        2,
    ):
        raise ValueError(
            "Invalid start_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {start_points.shape}"
        )
    if end_points.shape.rank not in (2, 3) or end_points.shape[-2:] != (4, 2):
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
    if len(images.shape) == 3:
        images = tf.expand_dims(images, axis=0)
        need_squeeze = True

    if len(start_points.shape) == 2:
        start_points = tf.expand_dims(start_points, axis=0)
    if len(end_points.shape) == 2:
        end_points = tf.expand_dims(end_points, axis=0)

    if data_format == "channels_first":
        images = tf.transpose(images, (0, 2, 3, 1))

    transform = compute_homography_matrix(start_points, end_points)
    if len(transform.shape) == 1:
        transform = tf.expand_dims(transform, axis=0)

    output = tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=tf.cast(transform, dtype=tf.float32),
        output_shape=tf.shape(images)[1:-1],
        fill_value=fill_value,
        interpolation=interpolation.upper(),
    )
    output = tf.ensure_shape(output, images.shape)

    if data_format == "channels_first":
        output = tf.transpose(output, (0, 3, 1, 2))
    if need_squeeze:
        output = tf.squeeze(output, axis=0)
    return output


def compute_homography_matrix(start_points, end_points):
    start_x1, start_y1 = start_points[:, 0, 0], start_points[:, 0, 1]
    start_x2, start_y2 = start_points[:, 1, 0], start_points[:, 1, 1]
    start_x3, start_y3 = start_points[:, 2, 0], start_points[:, 2, 1]
    start_x4, start_y4 = start_points[:, 3, 0], start_points[:, 3, 1]

    end_x1, end_y1 = end_points[:, 0, 0], end_points[:, 0, 1]
    end_x2, end_y2 = end_points[:, 1, 0], end_points[:, 1, 1]
    end_x3, end_y3 = end_points[:, 2, 0], end_points[:, 2, 1]
    end_x4, end_y4 = end_points[:, 3, 0], end_points[:, 3, 1]

    coefficient_matrix = tf.stack(
        [
            tf.stack(
                [
                    end_x1,
                    end_y1,
                    tf.ones_like(end_x1),
                    tf.zeros_like(end_x1),
                    tf.zeros_like(end_x1),
                    tf.zeros_like(end_x1),
                    -start_x1 * end_x1,
                    -start_x1 * end_y1,
                ],
                axis=-1,
            ),
            tf.stack(
                [
                    tf.zeros_like(end_x1),
                    tf.zeros_like(end_x1),
                    tf.zeros_like(end_x1),
                    end_x1,
                    end_y1,
                    tf.ones_like(end_x1),
                    -start_y1 * end_x1,
                    -start_y1 * end_y1,
                ],
                axis=-1,
            ),
            tf.stack(
                [
                    end_x2,
                    end_y2,
                    tf.ones_like(end_x2),
                    tf.zeros_like(end_x2),
                    tf.zeros_like(end_x2),
                    tf.zeros_like(end_x2),
                    -start_x2 * end_x2,
                    -start_x2 * end_y2,
                ],
                axis=-1,
            ),
            tf.stack(
                [
                    tf.zeros_like(end_x2),
                    tf.zeros_like(end_x2),
                    tf.zeros_like(end_x2),
                    end_x2,
                    end_y2,
                    tf.ones_like(end_x2),
                    -start_y2 * end_x2,
                    -start_y2 * end_y2,
                ],
                axis=-1,
            ),
            tf.stack(
                [
                    end_x3,
                    end_y3,
                    tf.ones_like(end_x3),
                    tf.zeros_like(end_x3),
                    tf.zeros_like(end_x3),
                    tf.zeros_like(end_x3),
                    -start_x3 * end_x3,
                    -start_x3 * end_y3,
                ],
                axis=-1,
            ),
            tf.stack(
                [
                    tf.zeros_like(end_x3),
                    tf.zeros_like(end_x3),
                    tf.zeros_like(end_x3),
                    end_x3,
                    end_y3,
                    tf.ones_like(end_x3),
                    -start_y3 * end_x3,
                    -start_y3 * end_y3,
                ],
                axis=-1,
            ),
            tf.stack(
                [
                    end_x4,
                    end_y4,
                    tf.ones_like(end_x4),
                    tf.zeros_like(end_x4),
                    tf.zeros_like(end_x4),
                    tf.zeros_like(end_x4),
                    -start_x4 * end_x4,
                    -start_x4 * end_y4,
                ],
                axis=-1,
            ),
            tf.stack(
                [
                    tf.zeros_like(end_x4),
                    tf.zeros_like(end_x4),
                    tf.zeros_like(end_x4),
                    end_x4,
                    end_y4,
                    tf.ones_like(end_x4),
                    -start_y4 * end_x4,
                    -start_y4 * end_y4,
                ],
                axis=-1,
            ),
        ],
        axis=1,
    )

    target_vector = tf.stack(
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
        axis=-1,
    )
    target_vector = tf.expand_dims(target_vector, axis=-1)

    homography_matrix = tf.linalg.solve(coefficient_matrix, target_vector)
    homography_matrix = tf.reshape(homography_matrix, [-1, 8])

    return homography_matrix


def _mirror_index_fixer(index, size):
    s = size - 1  # Half-wavelength of triangular wave
    # Scaled, integer-valued version of the triangular wave |x - round(x)|
    return tf.abs((index + s) % (2 * s) - s)


def _reflect_index_fixer(index, size):
    return tf.math.floordiv(
        _mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2
    )


_INDEX_FIXERS = {
    "constant": lambda index, size: index,
    "nearest": lambda index, size: tf.clip_by_value(index, 0, size - 1),
    "wrap": lambda index, size: index % size,
    "mirror": _mirror_index_fixer,
    "reflect": _reflect_index_fixer,
}


def _nearest_indices_and_weights(coordinate):
    coordinate = (
        coordinate if coordinate.dtype.is_integer else tf.round(coordinate)
    )
    index = tf.cast(coordinate, tf.int32)
    weight = tf.constant(1, coordinate.dtype)
    return [(index, weight)]


def _linear_indices_and_weights(coordinate):
    lower = tf.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = tf.cast(lower, tf.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0.0
):
    input_arr = convert_to_tensor(inputs)
    coordinate_arrs = convert_to_tensor(coordinates)

    if coordinate_arrs.shape[0] != len(input_arr.shape):
        raise ValueError(
            "First dim of `coordinates` must be the same as the rank of "
            "`inputs`. "
            f"Received inputs with shape: {input_arr.shape} and coordinate "
            f"leading dim of {coordinate_arrs.shape[0]}"
        )
    if len(coordinate_arrs.shape) < 2:
        raise ValueError(
            "Invalid coordinates rank: expected at least rank 2."
            f" Received input with shape: {coordinate_arrs.shape}"
        )

    fill_value = convert_to_tensor(fill_value, dtype=input_arr.dtype)

    coordinate_arrs = tf.unstack(coordinate_arrs, axis=0)

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    else:
        raise NotImplementedError("map_coordinates currently requires order<=1")

    def process_coordinates(coords, size):
        if fill_mode == "constant":
            valid = (coords >= 0) & (coords < size)
            safe_coords = tf.clip_by_value(coords, 0, size - 1)
            return safe_coords, valid
        elif fill_mode == "nearest":
            return tf.clip_by_value(coords, 0, size - 1), tf.ones_like(
                coords, dtype=tf.bool
            )
        elif fill_mode in ["mirror", "reflect"]:
            coords = tf.abs(coords)
            size_2 = size * 2
            mod = tf.math.mod(coords, size_2)
            under = mod < size
            over = ~under
            # reflect mode is same as mirror for under
            coords = tf.where(under, mod, size_2 - mod)
            # for reflect mode, adjust the over case
            if fill_mode == "reflect":
                coords = tf.where(over, coords - 1, coords)
            return coords, tf.ones_like(coords, dtype=tf.bool)
        elif fill_mode == "wrap":
            coords = tf.math.mod(coords, size)
            return coords, tf.ones_like(coords, dtype=tf.bool)
        else:
            raise ValueError(f"Unknown fill_mode: {fill_mode}")

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinate_arrs, input_arr.shape):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            safe_index, valid = process_coordinates(index, size)
            valid_interp.append((safe_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = zip(*items)
        indices = tf.transpose(tf.stack(indices))

        gathered = tf.transpose(tf.gather_nd(input_arr, indices))

        if fill_mode == "constant":
            all_valid = tf.reduce_all(validities)
            gathered = tf.where(all_valid, gathered, fill_value)

        contribution = gathered
        outputs.append(
            functools.reduce(operator.mul, weights)
            * tf.cast(contribution, weights[0].dtype)
        )

    result = functools.reduce(operator.add, outputs)

    if input_arr.dtype.is_integer:
        result = tf.round(result)
    return tf.cast(result, input_arr.dtype)


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    def _create_gaussian_kernel(kernel_size, sigma, num_channels, dtype):
        def _get_gaussian_kernel1d(size, sigma):
            x = tf.range(size, dtype=dtype) - (size - 1) / 2
            kernel1d = tf.exp(-0.5 * (x / sigma) ** 2)
            return kernel1d / tf.reduce_sum(kernel1d)

        def _get_gaussian_kernel2d(size, sigma):
            size = tf.cast(size, dtype)
            kernel1d_x = _get_gaussian_kernel1d(size[0], sigma[0])
            kernel1d_y = _get_gaussian_kernel1d(size[1], sigma[1])
            return tf.tensordot(kernel1d_y, kernel1d_x, axes=0)

        kernel = _get_gaussian_kernel2d(kernel_size, sigma)
        kernel = tf.reshape(kernel, (kernel_size[0], kernel_size[1], 1, 1))
        kernel = tf.tile(kernel, [1, 1, num_channels, 1])
        kernel = tf.cast(kernel, dtype)
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
    if len(images.shape) == 3:
        images = tf.expand_dims(images, axis=0)
        need_squeeze = True

    if data_format == "channels_first":
        images = tf.transpose(images, (0, 2, 3, 1))

    num_channels = tf.shape(images)[-1]
    kernel = _create_gaussian_kernel(kernel_size, sigma, num_channels, dtype)

    blurred_images = tf.nn.depthwise_conv2d(
        images, kernel, strides=[1, 1, 1, 1], padding="SAME"
    )

    if data_format == "channels_first":
        blurred_images = tf.transpose(blurred_images, (0, 3, 1, 2))
    if need_squeeze:
        blurred_images = tf.squeeze(blurred_images, axis=0)

    return blurred_images
