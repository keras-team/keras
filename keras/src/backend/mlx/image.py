import functools
import itertools
import operator

import mlx.core as mx

from keras.src import backend
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import to_mlx_dtype


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
    raise NotImplementedError("elastic_transform not yet implemented in mlx.")


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    raise NotImplementedError("perspective_transform not yet implemented in mlx.")


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    raise NotImplementedError("gaussian_blur not yet implemented in mlx.")