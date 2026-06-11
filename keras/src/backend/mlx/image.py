import functools
import itertools
import operator

import mlx.core as mx

from keras.src import backend
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import to_mlx_dtype
from keras.src.backend.mlx.random import mlx_draw_seed


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
    mlx_dtype = to_mlx_dtype(compute_dtype)
    images = images.astype(mlx_dtype)

    # Ref: tf.image.rgb_to_grayscale
    rgb_weights = convert_to_tensor(
        [0.2989, 0.5870, 0.1140], dtype=images.dtype
    )
    images = mx.tensordot(images, rgb_weights, axes=[[channels_axis], [-1]])
    images = mx.expand_dims(images, axis=channels_axis)
    return images.astype(original_dtype)


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
    eps = mx.finfo(dtype).min
    images = mx.where(mx.abs(images) < eps, 0.0, images)
    red, green, blue = mx.split(images, 3, channels_axis)
    red = mx.squeeze(red, channels_axis)
    green = mx.squeeze(green, channels_axis)
    blue = mx.squeeze(blue, channels_axis)

    def rgb_planes_to_hsv_planes(r, g, b):
        value = mx.maximum(mx.maximum(r, g), b)
        minimum = mx.minimum(mx.minimum(r, g), b)
        range_ = value - minimum

        safe_value = mx.where(value > 0, value, 1.0)
        safe_range = mx.where(range_ > 0, range_, 1.0)

        saturation = mx.where(value > 0, range_ / safe_value, 0.0)
        norm = 1.0 / (6.0 * safe_range)

        hue = mx.where(
            value == g,
            norm * (b - r) + 2.0 / 6.0,
            norm * (r - g) + 4.0 / 6.0,
        )
        hue = mx.where(value == r, norm * (g - b), hue)
        hue = mx.where(range_ > 0, hue, 0.0) + (hue < 0.0).astype(hue.dtype)
        return hue, saturation, value

    images = mx.stack(
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
    hue, saturation, value = mx.split(images, 3, channels_axis)
    hue = mx.squeeze(hue, channels_axis)
    saturation = mx.squeeze(saturation, channels_axis)
    value = mx.squeeze(value, channels_axis)

    def hsv_planes_to_rgb_planes(hue, saturation, value):
        dh = (hue % 1.0) * 6.0
        dr = mx.clip(mx.abs(dh - 3.0) - 1.0, 0.0, 1.0)
        dg = mx.clip(2.0 - mx.abs(dh - 2.0), 0.0, 1.0)
        db = mx.clip(2.0 - mx.abs(dh - 4.0), 0.0, 1.0)
        one_minus_s = 1.0 - saturation

        red = value * (one_minus_s + saturation * dr)
        green = value * (one_minus_s + saturation * dg)
        blue = value * (one_minus_s + saturation * db)
        return red, green, blue

    images = mx.stack(
        hsv_planes_to_rgb_planes(hue, saturation, value), axis=channels_axis
    )
    return images


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
            contribution = mx.where(all_valid, input_arr[indices], fill_value)
        outputs.append(functools.reduce(operator.mul, weights) * contribution)
    result = functools.reduce(operator.add, outputs)
    if _is_integer(input_arr):
        result = result if _is_integer(result) else mx.round(result)
    return result.astype(input_arr.dtype)


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

    images = convert_to_tensor(images)
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

    # unbatched case
    need_squeeze = False
    if len(images.shape) == 3:
        images = mx.expand_dims(images, axis=0)
        need_squeeze = True
    if len(transform.shape) == 1:
        transform = mx.expand_dims(transform, axis=0)

    if data_format == "channels_first":
        images = mx.transpose(images, (0, 2, 3, 1))

    batch_size = images.shape[0]

    # get indices
    meshgrid = mx.meshgrid(
        *[mx.arange(size) for size in images.shape[1:]], indexing="ij"
    )
    indices = mx.concatenate(
        [mx.expand_dims(x, axis=-1) for x in meshgrid], axis=-1
    )
    indices = mx.tile(indices, (batch_size, 1, 1, 1, 1))

    # swap the values
    a0 = transform[:, 0]
    a2 = transform[:, 2]
    b1 = transform[:, 4]
    b2 = transform[:, 5]
    transform[:, 0] = b1
    transform[:, 2] = b2
    transform[:, 4] = a0
    transform[:, 5] = a2

    # deal with transform
    transform = mx.pad(transform, pad_width=[[0, 0], [0, 1]], constant_values=1)
    transform = mx.reshape(transform, (batch_size, 3, 3))
    offset = transform[:, 0:2, 2]
    offset = mx.pad(offset, pad_width=[[0, 0], [0, 1]])
    transform[:, 0:2, 2] = 0

    # transform the indices
    coordinates = mx.einsum("Bhwij, Bjk -> Bhwik", indices, transform)
    coordinates = mx.moveaxis(coordinates, source=-1, destination=1)
    coordinates += offset.reshape((*offset.shape, 1, 1, 1))

    affined = mx.stack(
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
        affined = mx.transpose(affined, (0, 3, 1, 2))
    if need_squeeze:
        affined = mx.squeeze(affined, axis=0)
    return affined


def _resize_nearest(x, output_shape):
    # Ref: jax.image.resize
    input_shape = x.shape
    assert len(input_shape) == len(output_shape)
    spatial_dims = tuple(
        i for i in range(len(input_shape)) if input_shape[i] != output_shape[i]
    )
    for d in spatial_dims:
        m = input_shape[d]
        n = output_shape[d]
        offsets = (mx.arange(n, dtype=mx.float32) + 0.5) * m / n
        offsets = mx.floor(offsets.astype(mx.float32)).astype(mx.int32)
        indices = [slice(None)] * len(input_shape)
        indices[d] = offsets
        x = x[tuple(indices)]
    return x


def _fill_lanczos_kernel(radius, x):
    # Ref: jax.image.resize
    y = radius * mx.sin(mx.pi * x) * mx.sin(mx.pi * x / radius)
    #  out = y / (np.pi ** 2 * x ** 2) where x >1e-3, 1 otherwise
    out = mx.where(
        x > 1e-3, mx.divide(y, mx.where(x != 0, mx.pi**2 * x**2, 1.0)), 1.0
    )
    return mx.where(x > radius, 0.0, out)


def _fill_keys_cubic_kernel(x):
    # Ref: jax.image.resize
    # http://ieeexplore.ieee.org/document/1163711/
    # R. G. Keys. Cubic convolution interpolation for digital image processing.
    # IEEE Transactions on Acoustics, Speech, and Signal Processing,
    # 29(6):1153–1160, 1981.
    out = ((1.5 * x - 2.5) * x) * x + 1.0
    out = mx.where(x >= 1.0, ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0, out)
    return mx.where(x >= 2.0, 0.0, out)


def _fill_triangle_kernel(x):
    # Ref: jax.image.resize
    return mx.maximum(0, 1 - mx.abs(x))


RESIZE_INTERPOLATIONS = {
    "bilinear": _fill_triangle_kernel,
    "nearest": None,
    "lanczos3": lambda x: _fill_lanczos_kernel(3.0, x),
    "lanczos5": lambda x: _fill_lanczos_kernel(5.0, x),
    "bicubic": _fill_keys_cubic_kernel,
}


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
    if pad_to_aspect_ratio and crop_to_aspect_ratio:
        raise ValueError(
            "Only one of `pad_to_aspect_ratio` & `crop_to_aspect_ratio` "
            "can be `True`."
        )
    if interpolation not in RESIZE_INTERPOLATIONS.keys():
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{set(RESIZE_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )
    if not len(size) == 2:
        raise ValueError(
            "Argument `size` must be a tuple of two elements "
            f"(height, width). Received: size={size}"
        )
    if fill_mode != "constant":
        raise ValueError(
            "Invalid value for argument `fill_mode`. Only `'constant'` "
            f"is supported. Received: fill_mode={fill_mode}"
        )
    target_height, target_width = size
    size = tuple(size)
    images = convert_to_tensor(images)

    if images.ndim not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    # convert images to shape (b, h, w, c)
    need_squeeze = False
    if images.ndim == 3:
        images = mx.expand_dims(images, axis=0)
        need_squeeze = True
    if data_format == "channels_first":
        images = mx.transpose(images, axes=(0, 2, 3, 1))
    batch_size = images.shape[0]
    channels = images.shape[-1]

    if crop_to_aspect_ratio:
        shape = images.shape
        height, width = shape[-3], shape[-2]
        crop_height = int(float(width * target_height) / target_width)
        crop_height = max(min(height, crop_height), 1)
        crop_width = int(float(height * target_width) / target_height)
        crop_width = max(min(width, crop_width), 1)
        crop_box_hstart = int(float(height - crop_height) / 2)
        crop_box_wstart = int(float(width - crop_width) / 2)
        images = images[
            :,
            crop_box_hstart : crop_box_hstart + crop_height,
            crop_box_wstart : crop_box_wstart + crop_width,
            :,
        ]
    elif pad_to_aspect_ratio:
        shape = images.shape
        height, width, channels = shape[-3], shape[-2], shape[-1]
        pad_height = int(float(width * target_height) / target_width)
        pad_height = max(height, pad_height)
        pad_width = int(float(height * target_width) / target_height)
        pad_width = max(width, pad_width)
        img_box_hstart = int(float(pad_height - height) / 2)
        img_box_wstart = int(float(pad_width - width) / 2)

        if img_box_hstart > 0:
            padded_img = mx.concatenate(
                [
                    mx.ones(
                        (batch_size, img_box_hstart, width, channels),
                        dtype=images.dtype,
                    )
                    * fill_value,
                    images,
                    mx.ones(
                        (batch_size, img_box_hstart, width, channels),
                        dtype=images.dtype,
                    )
                    * fill_value,
                ],
                axis=1,
            )
        elif img_box_wstart > 0:
            padded_img = mx.concatenate(
                [
                    mx.ones(
                        (batch_size, height, img_box_wstart, channels),
                        dtype=images.dtype,
                    )
                    * fill_value,
                    images,
                    mx.ones(
                        (batch_size, height, img_box_wstart, channels),
                        dtype=images.dtype,
                    )
                    * fill_value,
                ],
                axis=2,
            )
        else:
            padded_img = images

        images = padded_img

    # Ref: jax.image.resize
    output_shape = (batch_size, target_height, target_width, channels)
    if interpolation == "nearest":
        result = _resize_nearest(images, output_shape)
    elif interpolation in RESIZE_INTERPOLATIONS.keys():
        kernel = RESIZE_INTERPOLATIONS[interpolation]

        # this method assumes spatial dims are always at axes 1 and 2
        spatial_dims = (1, 2)
        scale = [
            1.0 if output_shape[d] == 0 else output_shape[d] / images.shape[d]
            for d in spatial_dims
        ]
        result = _scale_and_translate(
            images,
            output_shape,
            spatial_dims,
            scale,
            [0.0] * len(spatial_dims),
            kernel,
            antialias,
        )

    if data_format == "channels_first":
        result = mx.transpose(result, (0, 3, 1, 2))
    if need_squeeze:
        result = mx.squeeze(result, axis=0)
    return result


def _scale_and_translate(
    x,
    output_shape,
    spatial_dims,
    scale,
    translation,
    kernel,
    antialias,
):
    # Ref: jax.image.resize
    # the following shapes are always assumed
    B_in, H_in, W_in, C_in = x.shape
    B_out, H_out, W_out, C_out = output_shape
    if B_in != B_out:
        raise ValueError(
            "Invalid batch sizes: batch size in and out "
            f"should match. Received batch size in: {B_in}, "
            f"and batche size out: {B_out}"
        )
    if C_in != C_out:
        raise ValueError(
            "Invalid channels: in and out channels should "
            f"match. Received input channels: {C_in}, "
            f"and output channels: {C_out}"
        )

    w_mats = {}
    for i, d in enumerate(spatial_dims):
        if d == 1:  # height dimension
            w_mats[d] = _compute_weight_mat(
                H_in, H_out, scale[i], translation[i], kernel, antialias
            ).astype(x.dtype)
        elif d == 2:  # width dimension
            w_mats[d] = _compute_weight_mat(
                W_in, W_out, scale[i], translation[i], kernel, antialias
            ).astype(x.dtype)
        else:
            raise ValueError(f"Unexpected dimension {d} for 2D scaling.")

    w_h = w_mats[1]  # shape (H_in, H_out)
    w_w = w_mats[2]  # shape (W_in, W_out)

    new_x = mx.einsum("bhwc,hH,wW->bHWc", x, w_h, w_w)

    return new_x


def _compute_weight_mat(
    input_size, output_size, scale, translation, kernel, antialias
):
    # Ref: jax.image.resize
    dtype = mx.float32
    inv_scale = 1.0 / scale
    kernel_scale = mx.maximum(inv_scale, 1.0) if antialias else 1.0
    sample_f = (
        (mx.arange(output_size, dtype=dtype) + 0.5) * inv_scale
        - translation * inv_scale
        - 0.5
    )
    x = (
        mx.abs(
            sample_f[mx.newaxis, :]
            - mx.arange(input_size, dtype=dtype)[:, mx.newaxis]
        )
        / kernel_scale
    )
    weights = kernel(x)

    total_weight_sum = mx.sum(weights, axis=0, keepdims=True)
    eps = 1.1920928955078125e-07  # np.finfo(np.float32).eps
    weights = mx.where(
        mx.abs(total_weight_sum) > 1000.0 * float(eps),
        mx.divide(
            weights, mx.where(total_weight_sum != 0, total_weight_sum, 1.0)
        ),
        0.0,
    )
    input_size_minus_0_5 = input_size - 0.5
    return mx.where(
        mx.logical_and(sample_f >= -0.5, sample_f <= input_size_minus_0_5)[
            mx.newaxis, :
        ],
        weights,
        0,
    )


def compute_homography_matrix(start_points, end_points):
    # as implemented for the jax backend
    start_points = convert_to_tensor(start_points, dtype=mx.float32)
    end_points = convert_to_tensor(end_points, dtype=mx.float32)

    start_x, start_y = start_points[..., 0], start_points[..., 1]
    end_x, end_y = end_points[..., 0], end_points[..., 1]

    zeros = mx.zeros_like(end_x)
    ones = mx.ones_like(end_x)

    x_rows = mx.stack(
        [
            end_x,
            end_y,
            ones,
            zeros,
            zeros,
            zeros,
            -start_x * end_x,
            -start_x * end_y,
        ],
        axis=-1,
    )
    y_rows = mx.stack(
        [
            zeros,
            zeros,
            zeros,
            end_x,
            end_y,
            ones,
            -start_y * end_x,
            -start_y * end_y,
        ],
        axis=-1,
    )

    coefficient_matrix = mx.concatenate([x_rows, y_rows], axis=1)

    target_vector = mx.expand_dims(
        mx.concatenate([start_x, start_y], axis=-1), axis=-1
    )

    # solve the linear system: coefficient_matrix * homography = target_vector
    with mx.stream(mx.cpu):
        homography_matrix = mx.linalg.solve(coefficient_matrix, target_vector)

    return homography_matrix.squeeze(-1)


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    # perspective_transform based on implementation in jax backend
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS.keys():
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected one of "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    if start_points.shape[-2:] != (4, 2) or start_points.ndim not in (2, 3):
        raise ValueError(
            "Invalid start_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {start_points.shape}"
        )
    if end_points.shape[-2:] != (4, 2) or end_points.ndim not in (2, 3):
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

    images = convert_to_tensor(images)
    start_points = convert_to_tensor(start_points)
    end_points = convert_to_tensor(end_points)

    need_squeeze = False
    if len(images.shape) == 3:
        images = mx.expand_dims(images, axis=0)
        need_squeeze = True

    if len(start_points.shape) == 2:
        start_points = mx.expand_dims(start_points, axis=0)
    if len(end_points.shape) == 2:
        end_points = mx.expand_dims(end_points, axis=0)

    if data_format == "channels_first":
        images = mx.transpose(images, (0, 2, 3, 1))

    batch_size, height, width, channels = images.shape

    transforms = compute_homography_matrix(
        mx.array(start_points, dtype=mx.float32),
        mx.array(end_points, dtype=mx.float32),
    )

    x, y = mx.meshgrid(mx.arange(width), mx.arange(height), indexing="xy")
    grid = mx.stack(
        [x.flatten(), y.flatten(), mx.ones_like(x).flatten()], axis=0
    )

    outputs = []
    for b in range(batch_size):
        transform = transforms[b]

        # apply homography to grid coordinates
        denom = transform[6] * grid[0] + transform[7] * grid[1] + 1.0
        x_in = (
            transform[0] * grid[0] + transform[1] * grid[1] + transform[2]
        ) / denom
        y_in = (
            transform[3] * grid[0] + transform[4] * grid[1] + transform[5]
        ) / denom

        coords = mx.stack([y_in, x_in], axis=0)

        transformed = mx.zeros((height, width, channels), dtype=images.dtype)
        for c in range(channels):
            transformed_channel = map_coordinates(
                images[b, :, :, c],
                coords,
                order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                fill_mode="constant",
                fill_value=fill_value,
            ).reshape(height, width)

            transformed = transformed.at[:, :, c].add(transformed_channel)

        outputs.append(transformed)

    output = mx.stack(outputs, axis=0)

    if data_format == "channels_first":
        output = mx.transpose(output, (0, 3, 1, 2))
    if need_squeeze:
        output = mx.squeeze(output, axis=0)

    return output


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    # gaussian_blur similar to jax backend
    def _create_gaussian_kernel(kernel_size, sigma, dtype, num_channels):
        def _get_gaussian_kernel1d(size, sigma):
            x = mx.arange(size, dtype=dtype) - (size - 1) / 2
            kernel1d = mx.exp(-0.5 * (x / sigma) ** 2)
            return kernel1d / mx.sum(kernel1d)

        def _get_gaussian_kernel2d(size, sigma):
            kernel1d_x = _get_gaussian_kernel1d(size[0], sigma[0])
            kernel1d_y = _get_gaussian_kernel1d(size[1], sigma[1])
            return mx.outer(kernel1d_y, kernel1d_x)

        kernel2d = _get_gaussian_kernel2d(kernel_size, sigma)

        # mlx expects kernel with shape (C_out, spatial..., C_in)
        # for depthwise convolution with groups=C, we need (C, H, W, 1)
        kernel = kernel2d.reshape(1, kernel_size[0], kernel_size[1], 1)
        kernel = mx.tile(kernel, (num_channels, 1, 1, 1))

        return kernel

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    data_format = backend.standardize_data_format(data_format)
    images = convert_to_tensor(images)
    sigma = convert_to_tensor(sigma)
    dtype = images.dtype

    need_squeeze = False
    if images.ndim == 3:
        images = images[mx.newaxis, ...]
        need_squeeze = True

    if data_format == "channels_first":
        images = mx.transpose(images, (0, 2, 3, 1))

    num_channels = images.shape[-1]

    # mx.arange can only take integer input values
    kernel_size = tuple(int(k) for k in kernel_size)
    kernel = _create_gaussian_kernel(kernel_size, sigma, dtype, num_channels)

    # get padding for 'same' behavior
    pad_h = max(0, (kernel_size[0] - 1) // 2)
    pad_w = max(0, (kernel_size[1] - 1) // 2)
    padding = ((pad_h, pad_h), (pad_w, pad_w))

    blurred_images = mx.conv_general(
        images,
        kernel,
        stride=1,
        padding=padding,
        kernel_dilation=1,
        input_dilation=1,
        groups=num_channels,
        flip=False,
    )

    if data_format == "channels_first":
        blurred_images = mx.transpose(blurred_images, (0, 3, 1, 2))

    if need_squeeze:
        blurred_images = mx.squeeze(blurred_images, axis=0)

    return blurred_images


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
    # elastic_transform based on implementation in jax backend
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected one of "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )
    if fill_mode not in AFFINE_TRANSFORM_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected one of "
            f"{AFFINE_TRANSFORM_FILL_MODES}. Received: fill_mode={fill_mode}"
        )
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    images = convert_to_tensor(images)
    alpha = convert_to_tensor(alpha)
    sigma = convert_to_tensor(sigma)
    input_dtype = images.dtype
    kernel_size = (int(6 * sigma) | 1, int(6 * sigma) | 1)

    need_squeeze = False
    if len(images.shape) == 3:
        images = mx.expand_dims(images, axis=0)
        need_squeeze = True

    if data_format == "channels_last":
        batch_size, height, width, channels = images.shape
        channel_axis = -1
    else:
        batch_size, channels, height, width = images.shape
        channel_axis = 1

    mlx_seed = mlx_draw_seed(seed)
    if mlx_seed is not None:
        seed_dx, seed_dy = mx.random.split(mlx_seed)
    else:
        seed_dx, seed_dy = mlx_draw_seed(None), mlx_draw_seed(None)

    dx = mx.random.normal(
        shape=(batch_size, height, width),
        loc=0.0,
        scale=sigma,
        dtype=input_dtype,
        key=seed_dx,
    )

    dy = mx.random.normal(
        shape=(batch_size, height, width),
        loc=0.0,
        scale=sigma,
        dtype=input_dtype,
        key=seed_dy,
    )

    dx = gaussian_blur(
        mx.expand_dims(dx, axis=channel_axis),
        kernel_size=kernel_size,
        sigma=(sigma, sigma),
        data_format=data_format,
    )
    dy = gaussian_blur(
        mx.expand_dims(dy, axis=channel_axis),
        kernel_size=kernel_size,
        sigma=(sigma, sigma),
        data_format=data_format,
    )

    dx = mx.squeeze(dx, axis=channel_axis)
    dy = mx.squeeze(dy, axis=channel_axis)

    x_vals = mx.arange(width)
    y_vals = mx.arange(height)
    x, y = mx.meshgrid(x_vals, y_vals, indexing="xy")
    x = mx.expand_dims(x, axis=0)
    y = mx.expand_dims(y, axis=0)

    distorted_x = x + alpha * dx
    distorted_y = y + alpha * dy

    transformed_images = mx.zeros_like(images)
    if data_format == "channels_last":
        for i in range(channels):
            transformed_channel = []
            for b in range(batch_size):
                transformed_channel.append(
                    map_coordinates(
                        images[b, :, :, i],
                        [distorted_y[b], distorted_x[b]],
                        order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                        fill_mode=fill_mode,
                        fill_value=fill_value,
                    )
                )
            transformed_images = transformed_images.at[:, :, :, i].add(
                mx.stack(transformed_channel)
            )
    else:  # channels_first
        for i in range(channels):
            transformed_channel = []
            for b in range(batch_size):
                transformed_channel.append(
                    map_coordinates(
                        images[b, i, :, :],
                        [distorted_y[b], distorted_x[b]],
                        order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                        fill_mode=fill_mode,
                        fill_value=fill_value,
                    )
                )
            transformed_images = transformed_images.at[:, i, :, :].add(
                mx.stack(transformed_channel)
            )

    if need_squeeze:
        transformed_images = mx.squeeze(transformed_images, axis=0)

    transformed_images = transformed_images.astype(input_dtype)

    return transformed_images
