import mlx.core as mx
import ml_dtypes
import numpy as np

from keras.src import backend
from keras.src.backend.mlx.core import _mlx_dtype
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.numpy import image as _numpy_image

RESIZE_INTERPOLATIONS = (
    "bilinear",
    "nearest",
    "lanczos3",
    "lanczos5",
    "bicubic",
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
MAP_COORDINATES_FILL_MODES = {
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
}
SCALE_AND_TRANSLATE_METHODS = {
    "linear",
    "bilinear",
    "trilinear",
    "cubic",
    "bicubic",
    "tricubic",
    "lanczos3",
    "lanczos5",
}


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
    if images.shape[channels_axis] not in (1, 3):
        raise ValueError(
            "Invalid channel size: expected 3 (RGB) or 1 (Grayscale). "
            f"Received input with shape: images.shape={images.shape}"
        )
    if images.shape[channels_axis] == 1:
        return mx.array(images)
    # Convert to floats
    original_dtype = images.dtype
    compute_dtype = backend.result_type(images.dtype, float)
    images = images.astype(_mlx_dtype(compute_dtype))

    # Ref: tf.image.rgb_to_grayscale
    rgb_weights = mx.array(
        [0.2989, 0.5870, 0.1140], dtype=images.dtype
    )
    # mlx tensordot wants axes as [[a_axes], [b_axes]] (no tuples).
    grayscales = mx.tensordot(
        images, rgb_weights, axes=[[channels_axis], [-1]]
    )
    grayscales = mx.expand_dims(grayscales, axis=channels_axis)
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
    # `ml_dtypes.finfo` (not `np.finfo`) so bfloat16 is accepted.
    eps = ml_dtypes.finfo(dtype).eps
    images = mx.where(mx.abs(images) < eps, 0.0, images)
    red, green, blue = mx.split(
        images, [1, 2], axis=channels_axis
    )
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
        hue = mx.where(range_ > 0, hue, 0.0) + (
            hue < 0.0
        ).astype(hue.dtype)
        return hue, saturation, value

    images = mx.stack(
        rgb_planes_to_hsv_planes(red, green, blue), axis=channels_axis
    )
    return images.astype(_mlx_dtype(dtype))


def hsv_to_rgb(images, data_format=None):
    # Ref: dm pix
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
    if not backend.is_float_dtype(backend.standardize_dtype(dtype)):
        raise ValueError(
            "Invalid images dtype: expected float dtype. "
            f"Received: images.dtype={backend.standardize_dtype(dtype)}"
        )
    hue, saturation, value = mx.split(
        images, [1, 2], axis=channels_axis
    )
    hue = mx.squeeze(hue, channels_axis)
    saturation = mx.squeeze(saturation, channels_axis)
    value = mx.squeeze(value, channels_axis)

    def hsv_planes_to_rgb_planes(hue, saturation, value):
        dh = mx.remainder(hue, 1.0) * 6.0
        dr = mx.clip(mx.abs(dh - 3.0) - 1.0, 0.0, 1.0)
        dg = mx.clip(2.0 - mx.abs(dh - 2.0), 0.0, 1.0)
        db = mx.clip(2.0 - mx.abs(dh - 4.0), 0.0, 1.0)
        one_minus_s = 1.0 - saturation

        red = value * (one_minus_s + saturation * dr)
        green = value * (one_minus_s + saturation * dg)
        blue = value * (one_minus_s + saturation * db)
        return red, green, blue

    images = mx.stack(
        hsv_planes_to_rgb_planes(hue, saturation, value),
        axis=channels_axis,
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
    images = convert_to_tensor(images)
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
            if img_box_hstart > 0:
                if len(images.shape) == 4:
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
                else:
                    padded_img = mx.concatenate(
                        [
                            mx.ones(
                                (img_box_hstart, width, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                            images,
                            mx.ones(
                                (img_box_hstart, width, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                        ],
                        axis=0,
                    )
            elif img_box_wstart > 0:
                if len(images.shape) == 4:
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
                    padded_img = mx.concatenate(
                        [
                            mx.ones(
                                (height, img_box_wstart, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                            images,
                            mx.ones(
                                (height, img_box_wstart, channels),
                                dtype=images.dtype,
                            )
                            * fill_value,
                        ],
                        axis=1,
                    )
            else:
                padded_img = images
        else:
            if img_box_hstart > 0:
                if len(images.shape) == 4:
                    padded_img = mx.concatenate(
                        [
                            mx.ones(
                                (batch_size, channels, img_box_hstart, width),
                                dtype=images.dtype,
                            )
                            * fill_value,
                            images,
                            mx.ones(
                                (batch_size, channels, img_box_hstart, width),
                                dtype=images.dtype,
                            )
                            * fill_value,
                        ],
                        axis=2,
                    )
                else:
                    padded_img = mx.concatenate(
                        [
                            mx.ones((channels, img_box_hstart, width))
                            * fill_value,
                            images,
                            mx.ones((channels, img_box_hstart, width))
                            * fill_value,
                        ],
                        axis=1,
                    )
            elif img_box_wstart > 0:
                if len(images.shape) == 4:
                    padded_img = mx.concatenate(
                        [
                            mx.ones(
                                (batch_size, channels, height, img_box_wstart),
                                dtype=images.dtype,
                            )
                            * fill_value,
                            images,
                            mx.ones(
                                (batch_size, channels, height, img_box_wstart),
                                dtype=images.dtype,
                            )
                            * fill_value,
                        ],
                        axis=3,
                    )
                else:
                    padded_img = mx.concatenate(
                        [
                            mx.ones((channels, height, img_box_wstart))
                            * fill_value,
                            images,
                            mx.ones((channels, height, img_box_wstart))
                            * fill_value,
                        ],
                        axis=2,
                    )
            else:
                padded_img = images
        images = padded_img

    return _resize(images, size, method=interpolation, antialias=antialias)


def _compute_weight_mat(
    input_size, output_size, scale, translation, kernel, antialias
):
    # Compute the resampling weight matrix exactly like the numpy backend
    # (in float64, on the CPU) so the result matches the reference bit-for-bit.
    # The matrix is small (input_size x output_size); the heavy resampling
    # tensordot is then run natively on the MLX graph.
    dtype = np.result_type(scale, translation)
    inv_scale = 1.0 / scale
    kernel_scale = np.maximum(inv_scale, 1.0) if antialias else 1.0

    sample_f = (
        (np.arange(output_size, dtype=dtype) + 0.5) * inv_scale
        - translation * inv_scale
        - 0.5
    )

    x = (
        np.abs(
            sample_f[np.newaxis, :]
            - np.arange(input_size, dtype=dtype)[:, np.newaxis]
        )
        / kernel_scale
    )

    weights = kernel(x)

    total_weight_sum = np.sum(weights, axis=0, keepdims=True)
    weights = np.where(
        np.abs(total_weight_sum) > 1000.0 * np.finfo(np.float32).eps,
        np.divide(
            weights, np.where(total_weight_sum != 0, total_weight_sum, 1)
        ),
        0,
    )

    input_size_minus_0_5 = input_size - 0.5
    return np.where(
        np.logical_and(sample_f >= -0.5, sample_f <= input_size_minus_0_5)[
            np.newaxis, :
        ],
        weights,
        0,
    )


def _resize(image, shape, method, antialias):
    if method == "nearest":
        return _resize_nearest(image, shape)
    else:
        kernel = _kernels.get(method, None)
    if kernel is None:
        raise ValueError("Unknown resize method")

    spatial_dims = tuple(
        i for i in range(len(shape)) if image.shape[i] != shape[i]
    )
    scale = [
        shape[d] / image.shape[d] if image.shape[d] != 0 else 1.0
        for d in spatial_dims
    ]

    return _scale_and_translate(
        image,
        shape,
        spatial_dims,
        scale,
        [0.0] * len(spatial_dims),
        kernel,
        antialias,
    )


def _resize_nearest(x, output_shape):
    input_shape = x.shape
    spatial_dims = tuple(
        i for i in range(len(input_shape)) if input_shape[i] != output_shape[i]
    )

    for d in spatial_dims:
        m, n = input_shape[d], output_shape[d]
        offsets = (np.arange(n, dtype=np.float32) + 0.5) * m / n
        offsets = np.floor(offsets).astype(np.int32)
        indices = [slice(None)] * len(input_shape)
        indices[d] = mx.array(offsets)
        x = x[tuple(indices)]
    return x


def _fill_triangle_kernel(x):
    return np.maximum(0, 1 - np.abs(x))


def _fill_keys_cubic_kernel(x):
    out = ((1.5 * x - 2.5) * x) * x + 1.0
    out = np.where(x >= 1.0, ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0, out)
    return np.where(x >= 2.0, 0.0, out)


def _fill_lanczos_kernel(radius, x):
    y = radius * np.sin(np.pi * x) * np.sin(np.pi * x / radius)
    out = np.where(
        x > 1e-3, np.divide(y, np.where(x != 0, np.pi**2 * x**2, 1)), 1
    )
    return np.where(x > radius, 0.0, out)


_kernels = {
    "linear": _fill_triangle_kernel,
    "bilinear": _fill_triangle_kernel,  # For `resize`.
    "cubic": _fill_keys_cubic_kernel,
    "bicubic": _fill_keys_cubic_kernel,  # For `resize`.
    "lanczos3": lambda x: _fill_lanczos_kernel(3.0, x),
    "lanczos5": lambda x: _fill_lanczos_kernel(5.0, x),
}


def _scale_and_translate(
    x, output_shape, spatial_dims, scale, translation, kernel, antialias
):
    input_shape = x.shape

    if len(spatial_dims) == 0:
        return x

    if not backend.is_float_dtype(backend.standardize_dtype(x.dtype)):
        output = x.astype(mx.float32)
        use_rounding = True
    else:
        output = mx.array(x)
        use_rounding = False

    for i, d in enumerate(spatial_dims):
        d = d % x.ndim
        m, n = input_shape[d], output_shape[d]

        w = _compute_weight_mat(
            m, n, scale[i], translation[i], kernel, antialias
        )
        # Bring the (small) weight matrix onto the MLX graph at the output
        # dtype, then run the resampling contraction natively.
        w = mx.array(w.astype(_np_dtype(output.dtype)))
        output = mx.tensordot(output, w, axes=[[d], [0]])
        output = mx.moveaxis(output, -1, d)

    if use_rounding:
        output = mx.clip(
            mx.round(output), mx.min(x), mx.max(x)
        )
        output = output.astype(x.dtype)
    return output


def affine_transform(
    images,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format=None,
):
    # Delegated to the numpy backend, which implements this via scipy. The
    # sampling/interpolation reference used by the test suite is scipy, so this
    # guarantees identical results; mlx tensors are marshalled at the boundary.
    images = convert_to_tensor(images)
    transform = convert_to_tensor(transform)
    result = _numpy_image.affine_transform(
        backend.convert_to_numpy(images),
        backend.convert_to_numpy(transform),
        interpolation=interpolation,
        fill_mode=fill_mode,
        fill_value=fill_value,
        data_format=data_format,
    )
    return convert_to_tensor(result, dtype=images.dtype)


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    images = convert_to_tensor(images)
    start_points = convert_to_tensor(start_points)
    end_points = convert_to_tensor(end_points)
    result = _numpy_image.perspective_transform(
        backend.convert_to_numpy(images),
        backend.convert_to_numpy(start_points),
        backend.convert_to_numpy(end_points),
        interpolation=interpolation,
        fill_value=fill_value,
        data_format=data_format,
    )
    return convert_to_tensor(result, dtype=images.dtype)


def compute_homography_matrix(start_points, end_points):
    # Pure linear-algebra port of the numpy implementation; runs natively on
    # the MLX graph (mlx.linalg.solve).
    start_points = convert_to_tensor(start_points)
    end_points = convert_to_tensor(end_points)
    dtype = backend.result_type(start_points.dtype, end_points.dtype, float)
    # `mx.linalg.solve` lacks support for float16 and bfloat16.
    compute_dtype = backend.result_type(dtype, "float32")
    start_points = start_points.astype(_mlx_dtype(dtype))
    end_points = end_points.astype(_mlx_dtype(dtype))

    start_x1, start_y1 = start_points[:, 0, 0], start_points[:, 0, 1]
    start_x2, start_y2 = start_points[:, 1, 0], start_points[:, 1, 1]
    start_x3, start_y3 = start_points[:, 2, 0], start_points[:, 2, 1]
    start_x4, start_y4 = start_points[:, 3, 0], start_points[:, 3, 1]

    end_x1, end_y1 = end_points[:, 0, 0], end_points[:, 0, 1]
    end_x2, end_y2 = end_points[:, 1, 0], end_points[:, 1, 1]
    end_x3, end_y3 = end_points[:, 2, 0], end_points[:, 2, 1]
    end_x4, end_y4 = end_points[:, 3, 0], end_points[:, 3, 1]

    ones = mx.ones_like(end_x1)
    zeros = mx.zeros_like(end_x1)

    def row(ex, ey, sx, sy):
        return mx.stack(
            [
                ex,
                ey,
                ones,
                zeros,
                zeros,
                zeros,
                -sx * ex,
                -sx * ey,
            ],
            axis=-1,
        ), mx.stack(
            [
                zeros,
                zeros,
                zeros,
                ex,
                ey,
                ones,
                -sy * ex,
                -sy * ey,
            ],
            axis=-1,
        )

    rows = [
        row(end_x1, end_y1, start_x1, start_y1),
        row(end_x2, end_y2, start_x2, start_y2),
        row(end_x3, end_y3, start_x3, start_y3),
        row(end_x4, end_y4, start_x4, start_y4),
    ]
    # rows[k] is (row_a, row_b); stack the 8 rows along axis 1.
    coefficient_matrix = mx.stack(
        [rows[0][0], rows[0][1], rows[1][0], rows[1][1],
         rows[2][0], rows[2][1], rows[3][0], rows[3][1]],
        axis=1,
    )

    target_vector = mx.stack(
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
    target_vector = mx.expand_dims(target_vector, axis=-1)
    coefficient_matrix = coefficient_matrix.astype(_mlx_dtype(compute_dtype))
    target_vector = target_vector.astype(_mlx_dtype(compute_dtype))
    # `mx.linalg.solve` is CPU-only; run it on a CPU stream (and keep the
    # dependent reshape/cast inside the same context).
    with mx.stream(mx.Device(mx.DeviceType.cpu, 0)):
        homography_matrix = mx.linalg.solve(coefficient_matrix, target_vector)
        homography_matrix = mx.reshape(homography_matrix, [-1, 8])
        return homography_matrix.astype(_mlx_dtype(dtype))


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0.0
):
    inputs = convert_to_tensor(inputs)
    coordinates = [convert_to_tensor(c) for c in coordinates]
    # Delegated to the numpy backend (scipy.ndimage.map_coordinates); the test
    # suite's reference is scipy, so we marshal to numpy, sample, and return an
    # mlx tensor at the input dtype.
    result = _numpy_image.map_coordinates(
        backend.convert_to_numpy(inputs),
        [backend.convert_to_numpy(c) for c in coordinates],
        order=order,
        fill_mode=fill_mode,
        fill_value=fill_value,
    )
    return convert_to_tensor(result, dtype=inputs.dtype)


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    def _create_gaussian_kernel(kernel_size, sigma, num_channels, dtype):
        def _get_gaussian_kernel1d(size, sigma):
            x = np.arange(size, dtype=dtype) - (size - 1) / 2
            kernel1d = np.exp(-0.5 * (x / sigma) ** 2)
            return kernel1d / np.sum(kernel1d)

        def _get_gaussian_kernel2d(size, sigma):
            size = np.asarray(size, dtype)
            kernel1d_x = _get_gaussian_kernel1d(size[0], sigma[0])
            kernel1d_y = _get_gaussian_kernel1d(size[1], sigma[1])
            return np.outer(kernel1d_y, kernel1d_x)

        kernel = _get_gaussian_kernel2d(kernel_size, sigma)
        return kernel.astype(dtype)

    images = convert_to_tensor(images)
    kernel_size = backend.convert_to_numpy(
        convert_to_tensor(kernel_size)
    ).astype(np.int32)
    sigma = convert_to_tensor(sigma)
    input_dtype = backend.standardize_dtype(images.dtype)
    # Mirrors scipy.signal.convolve2d: compute in float32.
    compute_dtype = backend.result_type(input_dtype, "float32")
    images = images.astype(_mlx_dtype(compute_dtype))
    sigma = sigma.astype(_mlx_dtype(compute_dtype))

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    need_squeeze = False
    if len(images.shape) == 3:
        images = mx.expand_dims(images, axis=0)
        need_squeeze = True

    if data_format == "channels_first":
        images = mx.transpose(images, (0, 2, 3, 1))

    batch_size, height, width, num_channels = images.shape

    kernel = _create_gaussian_kernel(
        kernel_size, backend.convert_to_numpy(sigma), num_channels, np.float32
    )

    kernel_h, kernel_w = kernel.shape[0], kernel.shape[1]
    pad_h = (kernel_h - 1) // 2
    pad_h_after = kernel_h - 1 - pad_h
    pad_w = (kernel_w - 1) // 2
    pad_w_after = kernel_w - 1 - pad_w

    # Pad H,W with zeros then run a depthwise "valid" convolution, matching
    # scipy.signal.convolve2d(padded, kernel, mode="valid").
    pad_config = [
        (0, 0),
        (pad_h, pad_h_after),
        (pad_w, pad_w_after),
        (0, 0),
    ]
    padded = mx.pad(images, pad_config)

    # Depthwise conv weight: (num_channels, kH, kW, 1), same kernel per channel.
    weight = mx.array(
        np.broadcast_to(
            kernel[:, :, np.newaxis], (num_channels, kernel_h, kernel_w, 1)
        )
    )
    blurred_images = mx.conv_general(
        padded, weight, stride=(1, 1), padding=(0, 0), groups=num_channels
    )

    if data_format == "channels_first":
        blurred_images = mx.transpose(blurred_images, (0, 3, 1, 2))
    if need_squeeze:
        blurred_images = mx.squeeze(blurred_images, axis=0)
    return blurred_images.astype(_mlx_dtype(input_dtype))


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
    images = convert_to_tensor(images)
    alpha = convert_to_tensor(alpha)
    sigma = convert_to_tensor(sigma)
    # Delegated to the numpy backend (scipy-based gaussian blur + map
    # coordinates); reference behaviour matches the test suite.
    result = _numpy_image.elastic_transform(
        backend.convert_to_numpy(images),
        alpha=backend.convert_to_numpy(alpha),
        sigma=backend.convert_to_numpy(sigma),
        interpolation=interpolation,
        fill_mode=fill_mode,
        fill_value=fill_value,
        seed=seed,
        data_format=data_format,
    )
    return convert_to_tensor(result, dtype=images.dtype)


def scale_and_translate(
    images,
    output_shape,
    scale,
    translation,
    spatial_dims,
    method,
    antialias=True,
):
    if method not in SCALE_AND_TRANSLATE_METHODS:
        raise ValueError(
            "Invalid value for argument `method`. Expected of one "
            f"{SCALE_AND_TRANSLATE_METHODS}. Received: method={method}"
        )
    if method in ("linear", "bilinear", "trilinear", "triangle"):
        method = "linear"
    elif method in ("cubic", "bicubic", "tricubic"):
        method = "cubic"

    images = convert_to_tensor(images)
    scale = convert_to_tensor(scale)
    translation = convert_to_tensor(translation)
    kernel = _kernels[method]
    dtype = backend.result_type(scale.dtype, translation.dtype)
    scale = scale.astype(_mlx_dtype(dtype))
    translation = translation.astype(_mlx_dtype(dtype))
    return _scale_and_translate(
        images,
        output_shape,
        spatial_dims,
        backend.convert_to_numpy(scale).tolist(),
        backend.convert_to_numpy(translation).tolist(),
        kernel,
        antialias,
    )


def sobel_edges(images, data_format=None):
    # Delegated to the numpy backend (scipy.ndimage.sobel); the test suite's
    # reference is scipy.
    images = convert_to_tensor(images)
    result = _numpy_image.sobel_edges(
        backend.convert_to_numpy(images), data_format=data_format
    )
    return convert_to_tensor(result, dtype=images.dtype)


# Late import to avoid a circular dependency at module load time.
from keras.src.backend.mlx.core import _mlx_dtype  # noqa: E402


def _np_dtype(mx_dt):
    # Map an mlx dtype back to the numpy dtype used to seed weight matrices.
    name = str(mx_dt).split(".")[-1]
    return np.dtype(name)
