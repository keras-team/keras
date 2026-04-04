import openvino.opset15 as ov_opset
from openvino import Type

from keras.src import backend
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import get_ov_output

RESIZE_INTERPOLATIONS = {
    "bilinear": "linear",
    "nearest": "nearest",
    "bicubic": "cubic",
    "lanczos3": "cubic",
    "lanczos5": "cubic",
}


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


def resize(
    images,
    size,
    interpolation="bilinear",
    antialias=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    fill_mode="constant",
    fill_value=0.0,
    data_format="channels_last",
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in RESIZE_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{tuple(RESIZE_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
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

    target_height, target_width = tuple(size)
    images = get_ov_output(images)
    rank = len(images.get_partial_shape())
    if rank not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    shape = images.get_partial_shape()
    shape = [None if dim.is_dynamic else dim.get_length() for dim in shape]
    if data_format == "channels_last":
        height_axis, width_axis = (-3, -2)
    else:
        height_axis, width_axis = (-2, -1)
    height = shape[height_axis]
    width = shape[width_axis]

    if (crop_to_aspect_ratio or pad_to_aspect_ratio) and (
        height is None or width is None
    ):
        raise ValueError(
            "`crop_to_aspect_ratio` and `pad_to_aspect_ratio` require "
            "static image height and width with the OpenVINO backend."
        )

    if crop_to_aspect_ratio:
        crop_height = int(float(width * target_height) / target_width)
        crop_height = max(min(height, crop_height), 1)
        crop_width = int(float(height * target_width) / target_height)
        crop_width = max(min(width, crop_width), 1)
        crop_box_hstart = int(float(height - crop_height) / 2)
        crop_box_wstart = int(float(width - crop_width) / 2)

        images = OpenVINOKerasTensor(images)
        if data_format == "channels_last":
            if rank == 4:
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
            if rank == 4:
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
        images = get_ov_output(images)
    elif pad_to_aspect_ratio:
        pad_height = int(float(width * target_height) / target_width)
        pad_height = max(height, pad_height)
        pad_width = int(float(height * target_width) / target_height)
        pad_width = max(width, pad_width)
        img_box_hstart = int(float(pad_height - height) / 2)
        img_box_wstart = int(float(pad_width - width) / 2)

        pads_begin = [0] * rank
        pads_end = [0] * rank
        pads_begin[height_axis] = img_box_hstart
        pads_end[height_axis] = img_box_hstart
        pads_begin[width_axis] = img_box_wstart
        pads_end[width_axis] = img_box_wstart

        fill_value = ov_opset.constant(
            fill_value, images.get_element_type()
        ).output(0)
        pads_begin = ov_opset.constant(pads_begin, Type.i32).output(0)
        pads_end = ov_opset.constant(pads_end, Type.i32).output(0)
        images = ov_opset.pad(
            images, pads_begin, pads_end, "constant", fill_value
        ).output(0)

    axes = [height_axis % rank, width_axis % rank]
    size = ov_opset.constant([target_height, target_width], Type.i32).output(0)
    axes = ov_opset.constant(axes, Type.i32).output(0)

    original_type = images.get_element_type()
    supported_types = {
        Type.f32,
        Type.f16,
        Type.bf16,
        Type.i8,
        Type.u8,
        Type.i32,
        Type.i64,
    }
    should_round_before_cast = False
    if original_type == Type.u8 and interpolation != "nearest":
        images = ov_opset.convert(images, Type.f32).output(0)
        should_round_before_cast = True
    elif original_type not in supported_types:
        images = ov_opset.convert(images, Type.f32).output(0)

    interpolate_kwargs = {
        "mode": RESIZE_INTERPOLATIONS[interpolation],
        "shape_calculation_mode": "sizes",
        "antialias": antialias,
        "axes": axes,
    }
    if interpolation == "nearest":
        interpolate_kwargs["coordinate_transformation_mode"] = (
            "tf_half_pixel_for_nn"
        )
        interpolate_kwargs["nearest_mode"] = "simple"
    elif interpolation in ("bicubic", "lanczos3", "lanczos5"):
        interpolate_kwargs["coordinate_transformation_mode"] = "half_pixel"
        interpolate_kwargs["cube_coeff"] = -0.5
    else:
        interpolate_kwargs["coordinate_transformation_mode"] = "half_pixel"

    resized = ov_opset.interpolate(images, size, **interpolate_kwargs).output(0)

    if should_round_before_cast:
        resized = ov_opset.round(resized, "half_to_even").output(0)
        resized = ov_opset.clamp(resized, 0.0, 255.0).output(0)
    if resized.get_element_type() != original_type:
        resized = ov_opset.convert(resized, original_type).output(0)
    return OpenVINOKerasTensor(resized)


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
    raise NotImplementedError(
        "`gaussian_blur` is not supported with openvino backend"
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
