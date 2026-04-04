import openvino.opset15 as ov_opset
from openvino import Type

from keras.src import backend
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import get_ov_output

RESIZE_INTERPOLATIONS = {
    "bilinear": "linear",
    "nearest": "nearest",
    "bicubic": "cubic",
}
UNSUPPORTED_INTERPOLATIONS = (
    "lanczos3",
    "lanczos5",
)


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
    if interpolation in UNSUPPORTED_INTERPOLATIONS:
        raise ValueError(
            "Resizing with Lanczos interpolation is "
            "not supported by the OpenVINO backend. "
            f"Received: interpolation={interpolation}."
        )
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
            f"images.shape={images.get_partial_shape()}"
        )

    if data_format == "channels_last":
        height_axis, width_axis = (-3, -2)
    else:
        height_axis, width_axis = (-2, -1)

    def _gather_dim(shape_tensor, axis):
        axis_node = ov_opset.constant([axis], Type.i32).output(0)
        axis0 = ov_opset.constant(0, Type.i32).output(0)
        return ov_opset.gather(shape_tensor, axis_node, axis0).output(0)

    def _floor_div_int(numerator, denominator):
        numerator_f = ov_opset.convert(numerator, Type.f32).output(0)
        denominator_f = ov_opset.convert(denominator, Type.f32).output(0)
        quotient = ov_opset.divide(numerator_f, denominator_f).output(0)
        floored = ov_opset.floor(quotient).output(0)
        return ov_opset.convert(floored, Type.i32).output(0)

    def _concat_scalars(nodes):
        return ov_opset.concat(nodes, axis=0).output(0)

    shape_node = ov_opset.shape_of(images, Type.i32).output(0)
    height_axis_index = height_axis % rank
    width_axis_index = width_axis % rank
    height = _gather_dim(shape_node, height_axis_index)
    width = _gather_dim(shape_node, width_axis_index)

    target_height_node = ov_opset.constant([target_height], Type.i32).output(0)
    target_width_node = ov_opset.constant([target_width], Type.i32).output(0)
    one_i32 = ov_opset.constant([1], Type.i32).output(0)
    zero_i32 = ov_opset.constant([0], Type.i32).output(0)

    if crop_to_aspect_ratio:
        crop_height = _floor_div_int(
            ov_opset.multiply(width, target_height_node).output(0),
            target_width_node,
        )
        crop_height = ov_opset.minimum(height, crop_height).output(0)
        crop_height = ov_opset.maximum(one_i32, crop_height).output(0)

        crop_width = _floor_div_int(
            ov_opset.multiply(height, target_width_node).output(0),
            target_height_node,
        )
        crop_width = ov_opset.minimum(width, crop_width).output(0)
        crop_width = ov_opset.maximum(one_i32, crop_width).output(0)

        crop_box_hstart = _floor_div_int(
            ov_opset.subtract(height, crop_height).output(0),
            ov_opset.constant([2], Type.i32).output(0),
        )
        crop_box_wstart = _floor_div_int(
            ov_opset.subtract(width, crop_width).output(0),
            ov_opset.constant([2], Type.i32).output(0),
        )

        crop_box_hend = ov_opset.add(crop_box_hstart, crop_height).output(0)
        crop_box_wend = ov_opset.add(crop_box_wstart, crop_width).output(0)

        begin_parts = []
        end_parts = []
        begin_mask = [1] * rank
        end_mask = [1] * rank
        for axis in range(rank):
            if axis == height_axis_index:
                begin_parts.append(crop_box_hstart)
                end_parts.append(crop_box_hend)
                begin_mask[axis] = 0
                end_mask[axis] = 0
            elif axis == width_axis_index:
                begin_parts.append(crop_box_wstart)
                end_parts.append(crop_box_wend)
                begin_mask[axis] = 0
                end_mask[axis] = 0
            else:
                begin_parts.append(zero_i32)
                end_parts.append(zero_i32)

        images = ov_opset.strided_slice(
            data=images,
            begin=_concat_scalars(begin_parts),
            end=_concat_scalars(end_parts),
            strides=ov_opset.constant([1] * rank, Type.i32).output(0),
            begin_mask=begin_mask,
            end_mask=end_mask,
        ).output(0)
    elif pad_to_aspect_ratio:
        pad_height = _floor_div_int(
            ov_opset.multiply(width, target_height_node).output(0),
            target_width_node,
        )
        pad_height = ov_opset.maximum(height, pad_height).output(0)

        pad_width = _floor_div_int(
            ov_opset.multiply(height, target_width_node).output(0),
            target_height_node,
        )
        pad_width = ov_opset.maximum(width, pad_width).output(0)

        img_box_hstart = _floor_div_int(
            ov_opset.subtract(pad_height, height).output(0),
            ov_opset.constant([2], Type.i32).output(0),
        )
        img_box_wstart = _floor_div_int(
            ov_opset.subtract(pad_width, width).output(0),
            ov_opset.constant([2], Type.i32).output(0),
        )

        pads_begin_parts = []
        pads_end_parts = []
        for axis in range(rank):
            if axis == height_axis_index:
                pads_begin_parts.append(img_box_hstart)
                pads_end_parts.append(img_box_hstart)
            elif axis == width_axis_index:
                pads_begin_parts.append(img_box_wstart)
                pads_end_parts.append(img_box_wstart)
            else:
                pads_begin_parts.append(zero_i32)
                pads_end_parts.append(zero_i32)

        fill_value = ov_opset.constant(
            fill_value, images.get_element_type()
        ).output(0)
        images = ov_opset.pad(
            images,
            _concat_scalars(pads_begin_parts),
            _concat_scalars(pads_end_parts),
            "constant",
            fill_value,
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
    should_adjust_uint8_bicubic = False
    if original_type == Type.u8 and interpolation != "nearest":
        images = ov_opset.convert(images, Type.f32).output(0)
        should_round_before_cast = True
        if interpolation == "bicubic":
            should_adjust_uint8_bicubic = True
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
    elif interpolation == "bicubic":
        interpolate_kwargs["coordinate_transformation_mode"] = "half_pixel"
        interpolate_kwargs["cube_coeff"] = -0.5
    else:
        interpolate_kwargs["coordinate_transformation_mode"] = "half_pixel"

    resized = ov_opset.interpolate(images, size, **interpolate_kwargs).output(0)

    if should_round_before_cast:
        resized = ov_opset.round(resized, "half_to_even").output(0)
        if should_adjust_uint8_bicubic:
            # Match TensorFlow/OpenCV-style uint8 bicubic behavior more closely
            # by nudging non-extreme values toward mid-range before clamping.
            low_mask = ov_opset.logical_and(
                ov_opset.greater(resized, ov_opset.constant(0.0, Type.f32)),
                ov_opset.less(resized, ov_opset.constant(128.0, Type.f32)),
            ).output(0)
            high_mask = ov_opset.logical_and(
                ov_opset.greater_equal(
                    resized, ov_opset.constant(128.0, Type.f32)
                ),
                ov_opset.less(resized, ov_opset.constant(255.0, Type.f32)),
            ).output(0)
            plus_one = ov_opset.add(
                resized,
                ov_opset.convert(low_mask, Type.f32),
            ).output(0)
            resized = ov_opset.subtract(
                plus_one,
                ov_opset.convert(high_mask, Type.f32),
            ).output(0)
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
