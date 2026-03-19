import numpy as np
import openvino.opset15 as ov_opset
from openvino import Type

from keras.src import backend
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import get_ov_output


def rgb_to_grayscale(images, data_format=None):
    images = get_ov_output(images)
    data_format = backend.standardize_data_format(data_format)
    if images.get_partial_shape().rank not in (3, 4):
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

    if images.shape[channel_axis] == 3:
        original_type = images.get_element_type()
        rgb_weights = ov_opset.constant(
            [0.2989, 0.5870, 0.1140], dtype=original_type
        ).output(0)
        if data_format == "channels_first":
            rgb_weights = ov_opset.unsqueeze(rgb_weights, axes=[-2, -1]).output(
                0
            )
        grayscales = ov_opset.multiply(images, rgb_weights).output(0)
        grayscales = ov_opset.reduce_sum(
            grayscales, reduction_axes=[channel_axis]
        ).output(0)
        grayscales = ov_opset.unsqueeze(grayscales, axes=[channel_axis]).output(
            0
        )
        if grayscales.get_element_type() != original_type:
            # Type of grayscales may be changed after unsqueeze, so we need to
            # convert it back to the original type.
            grayscales = ov_opset.convert(grayscales, original_type).output(0)

    return OpenVINOKerasTensor(grayscales)


def rgb_to_hsv(images, data_format=None):
    dtype = images.dtype
    images = get_ov_output(images)
    ov_type = images.get_element_type()
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
    eps = ov_opset.constant(backend.epsilon(), dtype=ov_type).output(0)
    images = ov_opset.select(
        ov_opset.less(ov_opset.abs(images), eps),
        ov_opset.constant(0.0, dtype=ov_type),
        images,
    ).output(0)
    rgb_channels = ov_opset.split(images, axis=channels_axis, num_splits=3)
    r, g, b = (
        rgb_channels.output(0),
        rgb_channels.output(1),
        rgb_channels.output(2),
    )

    def rgb_planes_to_hsv_planes(r, g, b):
        value = ov_opset.maximum(ov_opset.maximum(r, g), b).output(0)
        minimum = ov_opset.minimum(ov_opset.minimum(r, g), b).output(0)
        range_ = ov_opset.subtract(value, minimum).output(0)

        safe_value = ov_opset.select(
            ov_opset.greater(value, ov_opset.constant(0.0, dtype=ov_type)),
            value,
            ov_opset.constant(1.0, dtype=ov_type),
        ).output(0)
        safe_range = ov_opset.select(
            ov_opset.greater(range_, ov_opset.constant(0.0, dtype=ov_type)),
            range_,
            ov_opset.constant(1.0, dtype=ov_type),
        ).output(0)

        saturation = ov_opset.select(
            ov_opset.greater(value, ov_opset.constant(0.0, dtype=ov_type)),
            ov_opset.divide(range_, safe_value),
            ov_opset.constant(0.0, dtype=ov_type),
        ).output(0)
        norm = ov_opset.divide(
            ov_opset.constant(1.0, dtype=ov_type),
            ov_opset.multiply(
                ov_opset.constant(6.0, dtype=ov_type), safe_range
            ),
        ).output(0)

        hue = ov_opset.select(
            ov_opset.equal(value, g),
            ov_opset.add(
                ov_opset.multiply(norm, ov_opset.subtract(b, r)),
                ov_opset.constant(2.0 / 6.0, dtype=ov_type),
            ),
            ov_opset.add(
                ov_opset.multiply(norm, ov_opset.subtract(r, g)),
                ov_opset.constant(4.0 / 6.0, dtype=ov_type),
            ),
        ).output(0)
        hue = ov_opset.select(
            ov_opset.equal(value, r),
            ov_opset.multiply(norm, ov_opset.subtract(g, b)),
            hue,
        ).output(0)
        hue = ov_opset.select(
            ov_opset.greater(range_, ov_opset.constant(0.0, dtype=ov_type)),
            hue,
            ov_opset.constant(0.0, dtype=ov_type),
        ).output(0)
        hue = ov_opset.add(
            hue,
            ov_opset.convert(
                ov_opset.less(hue, ov_opset.constant(0.0, dtype=ov_type)),
                ov_type,
            ),
        ).output(0)
        return hue, saturation, value

    images = ov_opset.concat(
        rgb_planes_to_hsv_planes(r, g, b), axis=channels_axis
    ).output(0)
    return OpenVINOKerasTensor(images)


def hsv_to_rgb(images, data_format=None):
    dtype = images.dtype
    images = get_ov_output(images)
    ov_type = images.get_element_type()
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
    hsv_channels = ov_opset.split(images, axis=channels_axis, num_splits=3)
    hue, saturation, value = (
        hsv_channels.output(0),
        hsv_channels.output(1),
        hsv_channels.output(2),
    )

    def hsv_planes_to_rgb_planes(hue, saturation, value):
        def channel_value(channel_delta, one_minus_saturation):
            return ov_opset.multiply(
                value,
                ov_opset.add(
                    one_minus_saturation,
                    ov_opset.multiply(saturation, channel_delta),
                ),
            )

        dh = ov_opset.multiply(
            ov_opset.mod(hue, ov_opset.constant(1.0, dtype=ov_type)),
            ov_opset.constant(6.0, dtype=ov_type),
        ).output(0)
        one_const = ov_opset.constant(1.0, dtype=ov_type).output(0)
        two_const = ov_opset.constant(2.0, dtype=ov_type).output(0)
        three_const = ov_opset.constant(3.0, dtype=ov_type).output(0)
        four_const = ov_opset.constant(4.0, dtype=ov_type).output(0)
        dr = ov_opset.subtract(
            ov_opset.abs(ov_opset.subtract(dh, three_const)), one_const
        ).output(0)
        dr = ov_opset.clamp(dr, 0.0, 1.0).output(0)
        dg = ov_opset.subtract(
            two_const, ov_opset.abs(ov_opset.subtract(dh, two_const))
        ).output(0)
        dg = ov_opset.clamp(dg, 0.0, 1.0).output(0)
        db = ov_opset.subtract(
            two_const, ov_opset.abs(ov_opset.subtract(dh, four_const))
        ).output(0)
        db = ov_opset.clamp(db, 0.0, 1.0).output(0)
        one_minus_saturation = ov_opset.subtract(one_const, saturation).output(
            0
        )

        red = channel_value(dr, one_minus_saturation)
        green = channel_value(dg, one_minus_saturation)
        blue = channel_value(db, one_minus_saturation)
        return red, green, blue

    images = ov_opset.concat(
        hsv_planes_to_rgb_planes(hue, saturation, value), axis=channels_axis
    ).output(0)
    return OpenVINOKerasTensor(images)


RESIZE_INTERPOLATIONS = (
    "bilinear",
    "nearest",
    "lanczos3",
    "lanczos5",
    "bicubic",
)

_INTERP_MODE_MAP = {
    "bilinear": "linear",
    "bicubic": "cubic",
    "nearest": "nearest",
}


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
    if interpolation not in RESIZE_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{RESIZE_INTERPOLATIONS}. Received: interpolation={interpolation}"
        )
    if interpolation in ("lanczos3", "lanczos5"):
        raise NotImplementedError(
            f"`resize` with interpolation={interpolation!r} is not supported "
            "with the openvino backend."
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
    data_format = backend.standardize_data_format(data_format)
    image = get_ov_output(image)
    ov_type = image.get_element_type()
    rank = image.get_partial_shape().rank.get_length()
    if rank not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={image.get_partial_shape()}"
        )

    need_squeeze = rank == 3
    if need_squeeze:
        image = ov_opset.unsqueeze(
            image, ov_opset.constant([0], Type.i64)
        ).output(0)

    if data_format == "channels_last":
        image = ov_opset.transpose(
            image,
            ov_opset.constant([0, 3, 1, 2], Type.i64),
        ).output(0)

    img_shape = ov_opset.shape_of(image).output(0)
    h_node = ov_opset.gather(
        img_shape,
        ov_opset.constant(2, Type.i64),
        ov_opset.constant(0, Type.i64),
    ).output(0)
    w_node = ov_opset.gather(
        img_shape,
        ov_opset.constant(3, Type.i64),
        ov_opset.constant(0, Type.i64),
    ).output(0)

    target_h, target_w = int(size[0]), int(size[1])

    def _unsq(x):
        return ov_opset.unsqueeze(x, ov_opset.constant(0, Type.i64)).output(0)

    if crop_to_aspect_ratio:
        crop_h_f = ov_opset.divide(
            ov_opset.multiply(
                ov_opset.convert(w_node, "f32").output(0),
                ov_opset.constant(float(target_h), Type.f32).output(0),
            ).output(0),
            ov_opset.constant(float(target_w), Type.f32).output(0),
        ).output(0)
        crop_h = ov_opset.convert(
            ov_opset.floor(crop_h_f).output(0), "i64"
        ).output(0)
        one_i64 = ov_opset.constant(1, Type.i64).output(0)
        crop_h = ov_opset.minimum(
            h_node, ov_opset.maximum(crop_h, one_i64).output(0)
        ).output(0)

        crop_w_f = ov_opset.divide(
            ov_opset.multiply(
                ov_opset.convert(h_node, "f32").output(0),
                ov_opset.constant(float(target_w), Type.f32).output(0),
            ).output(0),
            ov_opset.constant(float(target_h), Type.f32).output(0),
        ).output(0)
        crop_w = ov_opset.convert(
            ov_opset.floor(crop_w_f).output(0), "i64"
        ).output(0)
        crop_w = ov_opset.minimum(
            w_node, ov_opset.maximum(crop_w, one_i64).output(0)
        ).output(0)

        two_c = ov_opset.constant(2, Type.i64).output(0)
        h_start = ov_opset.divide(
            ov_opset.subtract(h_node, crop_h).output(0), two_c
        ).output(0)
        w_start = ov_opset.divide(
            ov_opset.subtract(w_node, crop_w).output(0), two_c
        ).output(0)

        zero_c = ov_opset.constant([0], Type.i64).output(0)

        h_start_1d = _unsq(h_start)
        w_start_1d = _unsq(w_start)
        crop_h_1d = _unsq(crop_h)
        crop_w_1d = _unsq(crop_w)

        starts = ov_opset.concat(
            [zero_c, zero_c, h_start_1d, w_start_1d], 0
        ).output(0)
        stops = ov_opset.concat(
            [
                ov_opset.constant([np.iinfo(np.int64).max], Type.i64).output(0),
                ov_opset.constant([np.iinfo(np.int64).max], Type.i64).output(0),
                ov_opset.add(h_start_1d, crop_h_1d).output(0),
                ov_opset.add(w_start_1d, crop_w_1d).output(0),
            ],
            0,
        ).output(0)
        steps = ov_opset.constant([1, 1, 1, 1], Type.i64).output(0)
        image = ov_opset.slice(image, starts, stops, steps).output(0)

    elif pad_to_aspect_ratio:
        pad_h_f = ov_opset.divide(
            ov_opset.multiply(
                ov_opset.convert(w_node, "f32").output(0),
                ov_opset.constant(float(target_h), Type.f32).output(0),
            ).output(0),
            ov_opset.constant(float(target_w), Type.f32).output(0),
        ).output(0)
        pad_h = ov_opset.convert(
            ov_opset.floor(pad_h_f).output(0), "i64"
        ).output(0)
        pad_h = ov_opset.maximum(h_node, pad_h).output(0)

        pad_w_f = ov_opset.divide(
            ov_opset.multiply(
                ov_opset.convert(h_node, "f32").output(0),
                ov_opset.constant(float(target_w), Type.f32).output(0),
            ).output(0),
            ov_opset.constant(float(target_h), Type.f32).output(0),
        ).output(0)
        pad_w = ov_opset.convert(
            ov_opset.floor(pad_w_f).output(0), "i64"
        ).output(0)
        pad_w = ov_opset.maximum(w_node, pad_w).output(0)

        two_c = ov_opset.constant(2, Type.i64).output(0)
        h_offset = ov_opset.divide(
            ov_opset.subtract(pad_h, h_node).output(0), two_c
        ).output(0)
        w_offset = ov_opset.divide(
            ov_opset.subtract(pad_w, w_node).output(0), two_c
        ).output(0)

        h_offset_1d = _unsq(h_offset)
        w_offset_1d = _unsq(w_offset)
        h_pad_end = ov_opset.subtract(
            ov_opset.subtract(pad_h, h_node).output(0), h_offset
        ).output(0)
        w_pad_end = ov_opset.subtract(
            ov_opset.subtract(pad_w, w_node).output(0), w_offset
        ).output(0)
        h_pad_end_1d = _unsq(h_pad_end)
        w_pad_end_1d = _unsq(w_pad_end)

        zero_1d = ov_opset.constant([0], Type.i64).output(0)
        pads_begin = ov_opset.concat(
            [zero_1d, zero_1d, h_offset_1d, w_offset_1d], 0
        ).output(0)
        pads_end = ov_opset.concat(
            [zero_1d, zero_1d, h_pad_end_1d, w_pad_end_1d], 0
        ).output(0)
        fill_const = ov_opset.constant(fill_value, ov_type).output(0)
        image = ov_opset.pad(
            image, pads_begin, pads_end, "constant", fill_const
        ).output(0)

    ov_interp_mode = _INTERP_MODE_MAP[interpolation]
    sizes_node = ov_opset.constant([target_h, target_w], Type.i32).output(0)
    axes_node = ov_opset.constant([2, 3], Type.i32).output(0)
    image = ov_opset.interpolate(
        image,
        sizes_node,
        ov_interp_mode,
        "sizes",
        antialias=antialias,
        axes=axes_node,
    ).output(0)

    if data_format == "channels_last":
        image = ov_opset.transpose(
            image,
            ov_opset.constant([0, 2, 3, 1], Type.i64),
        ).output(0)

    if need_squeeze:
        image = ov_opset.squeeze(
            image, ov_opset.constant([0], Type.i64)
        ).output(0)

    return OpenVINOKerasTensor(image)


def affine_transform(
    images,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format=None,
):
    raise NotImplementedError(
        "`affine_transform` is not supported with openvino backend"
    )


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    raise NotImplementedError(
        "`perspective_transform` is not supported with openvino backend"
    )


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0
):
    raise NotImplementedError(
        "`map_coordinates` is not supported with openvino backend"
    )


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    data_format = backend.standardize_data_format(data_format)
    images = get_ov_output(images)
    ov_type = images.get_element_type()

    if images.get_partial_shape().rank.get_length() not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.get_partial_shape()}"
        )

    need_squeeze = images.get_partial_shape().rank.get_length() == 3
    if need_squeeze:
        images = ov_opset.unsqueeze(
            images, ov_opset.constant([0], Type.i64)
        ).output(0)

    if data_format == "channels_last":
        images = ov_opset.transpose(
            images, ov_opset.constant([0, 3, 1, 2], Type.i64)
        ).output(0)

    # kernel_size[0]/sigma[0] → height (y); kernel_size[1]/sigma[1] → width (x)
    kh, kw = int(kernel_size[0]), int(kernel_size[1])
    sigma_h, sigma_w = float(sigma[0]), float(sigma[1])

    def _gaussian_kernel_1d(k, s):
        center = (k - 1) / 2.0
        rng = ov_opset.range(
            ov_opset.constant(0.0, ov_type),
            ov_opset.constant(float(k), ov_type),
            ov_opset.constant(1.0, ov_type),
            ov_type.get_type_name(),
        ).output(0)
        x = ov_opset.subtract(
            rng, ov_opset.constant(center, ov_type).output(0)
        ).output(0)
        xs = ov_opset.divide(x, ov_opset.constant(s, ov_type).output(0)).output(
            0
        )
        xs2 = ov_opset.multiply(xs, xs).output(0)
        earg = ov_opset.multiply(
            ov_opset.constant(-0.5, ov_type).output(0), xs2
        ).output(0)
        k1d = ov_opset.exp(earg).output(0)
        total = ov_opset.reduce_sum(
            k1d, ov_opset.constant([0], Type.i32), keep_dims=True
        ).output(0)
        return ov_opset.divide(k1d, total).output(0)

    k1d_h = _gaussian_kernel_1d(kh, sigma_h)
    k1d_w = _gaussian_kernel_1d(kw, sigma_w)

    kh2d = ov_opset.reshape(
        k1d_h, ov_opset.constant([kh, 1], Type.i64), False
    ).output(0)
    kw2d = ov_opset.reshape(
        k1d_w, ov_opset.constant([1, kw], Type.i64), False
    ).output(0)
    k2d = ov_opset.matmul(kh2d, kw2d, False, False).output(0)

    k5d = ov_opset.reshape(
        k2d, ov_opset.constant([1, 1, 1, kh, kw], Type.i64), False
    ).output(0)
    img_shape = ov_opset.shape_of(images).output(0)
    c_node = ov_opset.slice(
        img_shape,
        ov_opset.constant([1], Type.i64).output(0),
        ov_opset.constant([2], Type.i64).output(0),
        ov_opset.constant([1], Type.i64).output(0),
    ).output(0)
    ones_tail = ov_opset.constant([1, 1, 1, 1], Type.i64).output(0)
    tiles = ov_opset.concat([c_node, ones_tail], 0).output(0)
    kernel = ov_opset.tile(k5d, tiles).output(0)

    pad_h, pad_w = kh // 2, kw // 2
    blurred = ov_opset.group_convolution(
        images,
        kernel,
        [1, 1],
        [pad_h, pad_w],
        [pad_h, pad_w],
        [1, 1],
        "EXPLICIT",
    ).output(0)

    if data_format == "channels_last":
        blurred = ov_opset.transpose(
            blurred, ov_opset.constant([0, 2, 3, 1], Type.i64)
        ).output(0)

    if need_squeeze:
        blurred = ov_opset.squeeze(
            blurred, ov_opset.constant([0], Type.i64)
        ).output(0)

    return OpenVINOKerasTensor(blurred)


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
    raise NotImplementedError(
        "`elastic_transform` is not supported with openvino backend"
    )


def scale_and_translate(
    images,
    output_shape,
    scale,
    translation,
    spatial_dims,
    method,
    antialias=True,
):
    raise NotImplementedError(
        "`scale_and_translate` is not supported with openvino backend"
    )
