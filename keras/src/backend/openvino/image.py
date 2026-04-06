import itertools

import numpy as np
import openvino.opset15 as ov_opset
from openvino import Type

from keras.src import backend
from keras.src.backend.openvino.core import DTYPES_MAX
from keras.src.backend.openvino.core import DTYPES_MIN
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import cast
from keras.src.backend.openvino.core import convert_to_tensor
from keras.src.backend.openvino.core import get_ov_output

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


def _mirror_index_fixer(index, size_node):
    """size_node: OV i32 scalar node."""
    one_c = ov_opset.constant(1, dtype=Type.i32).output(0)
    two_c = ov_opset.constant(2, dtype=Type.i32).output(0)
    zero_c = ov_opset.constant(0, dtype=Type.i32).output(0)
    s = ov_opset.subtract(size_node, one_c).output(0)  # size - 1
    s2 = ov_opset.multiply(two_c, s).output(0)  # 2 * (size - 1)
    # Guard mod-by-zero when size == 1
    safe_s2 = ov_opset.maximum(s2, one_c).output(0)
    diff = ov_opset.subtract(
        ov_opset.mod(ov_opset.add(index, s), safe_s2), s
    ).output(0)
    # Use max(d, 0-d) instead of abs(d): abs on integers is unreliable
    # under OV's INFERENCE_PRECISION_HINT=f32 for certain graph topologies.
    neg_diff = ov_opset.subtract(zero_c, diff).output(0)
    result = ov_opset.maximum(diff, neg_diff).output(0)
    # When size == 1 all indices map to 0.
    size_is_one = ov_opset.equal(size_node, one_c).output(0)
    return ov_opset.select(size_is_one, zero_c, result).output(0)


def _reflect_index_fixer(index, size_node):
    """size_node: OV i32 scalar node."""
    # floor_divide(_mirror(2*index + 1, 2*size + 1) - 1, 2)
    two_c = ov_opset.constant(2, dtype=Type.i32).output(0)
    one_c = ov_opset.constant(1, dtype=Type.i32).output(0)
    idx2 = ov_opset.add(ov_opset.multiply(two_c, index), one_c).output(0)
    size2_plus1 = ov_opset.add(
        ov_opset.multiply(two_c, size_node), one_c
    ).output(0)
    mirrored = _mirror_index_fixer(idx2, size2_plus1)
    # integer divide by 2 (result is always >= 0, so truncated == floor)
    return ov_opset.divide(ov_opset.subtract(mirrored, one_c), two_c).output(0)


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0
):
    fill_modes = {"constant", "nearest", "wrap", "mirror", "reflect"}
    if fill_mode not in fill_modes:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected one of "
            f"{fill_modes}. Received: fill_mode={fill_mode}"
        )
    if order not in (0, 1):
        raise ValueError(
            "Invalid value for argument `order`. Expected one of "
            f"[0, 1]. Received: order={order}"
        )

    inputs = convert_to_tensor(inputs)
    coordinates = convert_to_tensor(coordinates)
    inputs_ov = get_ov_output(inputs)
    coords_ov = get_ov_output(coordinates)
    input_shape = inputs_ov.get_partial_shape()
    ndim = input_shape.rank.get_length()
    coords_shape = coords_ov.get_partial_shape()
    if coords_shape.rank.is_static and coords_shape.rank.get_length() < 2:
        raise ValueError(
            "Invalid coordinates rank: expected at least rank 2."
            f" Received input with shape: {tuple(coords_shape.to_shape())}"
        )
    if coords_shape[0].is_static and coords_shape[0].get_length() != ndim:
        raise ValueError(
            "First dim of `coordinates` must be the same as the rank of "
            "`inputs`. "
            f"Received inputs with shape: {tuple(input_shape.to_shape())} and "
            f"coordinate leading dim of {coords_shape[0].get_length()}"
        )
    ov_type = inputs_ov.get_element_type()

    # Coordinates must be float for arithmetic
    coords_ov = ov_opset.convert(coords_ov, Type.f32).output(0)

    # Split coordinates into per-dimension tensors, each shape [*output_shape]
    axis_0 = ov_opset.constant(0, dtype=Type.i32).output(0)
    coord_list = [
        ov_opset.gather(
            coords_ov,
            ov_opset.constant(i, dtype=Type.i32).output(0),
            axis_0,
        ).output(0)
        for i in range(ndim)
    ]

    input_shape_node = ov_opset.shape_of(
        inputs_ov, output_type=Type.i32
    ).output(0)
    size_nodes = [
        ov_opset.gather(
            input_shape_node,
            ov_opset.constant(i, dtype=Type.i32).output(0),
            axis_0,
        ).output(0)
        for i in range(ndim)
    ]

    def fix_index(index, size_node):
        if fill_mode in ("constant", "nearest"):
            zero = ov_opset.constant(0, dtype=Type.i32).output(0)
            size_m1 = ov_opset.subtract(
                size_node, ov_opset.constant(1, dtype=Type.i32)
            ).output(0)
            return ov_opset.minimum(
                ov_opset.maximum(index, zero), size_m1
            ).output(0)
        elif fill_mode == "wrap":
            return ov_opset.floor_mod(index, size_node).output(0)
        elif fill_mode == "mirror":
            return _mirror_index_fixer(index, size_node)
        else:  # reflect
            return _reflect_index_fixer(index, size_node)

    def is_valid(index, size_node):
        if fill_mode != "constant":
            return None
        return ov_opset.logical_and(
            ov_opset.greater_equal(index, ov_opset.constant(0, dtype=Type.i32)),
            ov_opset.less(index, size_node),
        ).output(0)

    # Build per-dimension interpolation nodes
    interp_nodes_per_dim = []
    for i, coord in enumerate(coord_list):
        size_node = size_nodes[i]
        if order == 0:
            idx = ov_opset.convert(
                ov_opset.round(coord, mode="half_to_even"), Type.i32
            ).output(0)
            interp_nodes_per_dim.append(
                [(fix_index(idx, size_node), is_valid(idx, size_node), None)]
            )
        else:
            lower_f = ov_opset.floor(coord).output(0)
            upper_w = ov_opset.subtract(coord, lower_f).output(0)
            lower_w = ov_opset.subtract(
                ov_opset.constant(1.0, dtype=Type.f32), upper_w
            ).output(0)
            lower_idx = ov_opset.convert(lower_f, Type.i32).output(0)
            upper_idx = ov_opset.add(
                lower_idx, ov_opset.constant(1, dtype=Type.i32)
            ).output(0)
            interp_nodes_per_dim.append(
                [
                    (
                        fix_index(lower_idx, size_node),
                        is_valid(lower_idx, size_node),
                        lower_w,
                    ),
                    (
                        fix_index(upper_idx, size_node),
                        is_valid(upper_idx, size_node),
                        upper_w,
                    ),
                ]
            )

    fill_const = ov_opset.convert(
        ov_opset.constant(float(fill_value), dtype=Type.f32), ov_type
    ).output(0)

    output = None
    for items in itertools.product(*interp_nodes_per_dim):
        indices, validities, weights = zip(*items)

        # Stack indices: [*output_shape, ndim] for gather_nd
        stacked = ov_opset.concat(
            [ov_opset.unsqueeze(idx, axes=[-1]).output(0) for idx in indices],
            axis=-1,
        ).output(0)
        gathered = ov_opset.gather_nd(inputs_ov, stacked).output(0)

        if fill_mode == "constant":
            valid = validities[0]
            for v in validities[1:]:
                valid = ov_opset.logical_and(valid, v).output(0)
            gathered = ov_opset.select(valid, gathered, fill_const).output(0)

        if any(w is not None for w in weights):
            result_type = ov_type if not ov_type.is_integral() else Type.f32
            contribution = ov_opset.convert(gathered, result_type).output(0)
            for w in weights:
                if w is not None:
                    contribution = ov_opset.multiply(
                        contribution,
                        ov_opset.convert(w, result_type).output(0),
                    ).output(0)
        else:
            contribution = gathered

        output = (
            contribution
            if output is None
            else ov_opset.add(output, contribution).output(0)
        )

    if ov_type.is_integral() and order == 1:
        output = ov_opset.convert(
            ov_opset.round(
                ov_opset.convert(output, Type.f32).output(0),
                mode="half_to_even",
            ).output(0),
            ov_type,
        ).output(0)

    return OpenVINOKerasTensor(output)


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


def _ov_fill_triangle_kernel(x):
    t = x.get_element_type()
    # x is always non-negative; clamp(1 - x, 0, 1) == max(0, 1 - |x|)
    return ov_opset.clamp(
        ov_opset.subtract(ov_opset.constant(1.0, dtype=t), x), 0.0, 1.0
    ).output(0)


def _ov_fill_keys_cubic_kernel(x):
    t = x.get_element_type()
    c = lambda v: ov_opset.constant(v, dtype=t)
    # out = ((1.5x - 2.5)x)x + 1.0
    out = ov_opset.add(
        ov_opset.multiply(
            ov_opset.multiply(
                ov_opset.subtract(ov_opset.multiply(c(1.5), x), c(2.5)), x
            ),
            x,
        ),
        c(1.0),
    ).output(0)
    # out2 = ((-0.5x + 2.5)x - 4.0)x + 2.0
    out2 = ov_opset.add(
        ov_opset.multiply(
            ov_opset.subtract(
                ov_opset.multiply(
                    ov_opset.add(ov_opset.multiply(c(-0.5), x), c(2.5)), x
                ),
                c(4.0),
            ),
            x,
        ),
        c(2.0),
    ).output(0)
    out = ov_opset.select(ov_opset.greater_equal(x, c(1.0)), out2, out).output(
        0
    )
    return ov_opset.select(
        ov_opset.greater_equal(x, c(2.0)), c(0.0), out
    ).output(0)


def _ov_fill_lanczos_kernel(radius, x):
    t = x.get_element_type()
    c = lambda v: ov_opset.constant(v, dtype=t)
    pi_x = ov_opset.multiply(c(float(np.pi)), x).output(0)
    y = ov_opset.multiply(
        c(float(radius)),
        ov_opset.multiply(
            ov_opset.sin(pi_x),
            ov_opset.sin(ov_opset.divide(pi_x, c(float(radius)))),
        ),
    ).output(0)
    safe_denom = ov_opset.select(
        ov_opset.equal(x, c(0.0)),
        c(1.0),
        ov_opset.multiply(c(float(np.pi**2)), ov_opset.multiply(x, x)),
    ).output(0)
    out = ov_opset.select(
        ov_opset.greater(x, c(1e-3)), ov_opset.divide(y, safe_denom), c(1.0)
    ).output(0)
    return ov_opset.select(
        ov_opset.greater(x, c(float(radius))), c(0.0), out
    ).output(0)


_ov_kernels = {
    "linear": _ov_fill_triangle_kernel,
    "cubic": _ov_fill_keys_cubic_kernel,
    "lanczos3": lambda x: _ov_fill_lanczos_kernel(3.0, x),
    "lanczos5": lambda x: _ov_fill_lanczos_kernel(5.0, x),
}


def _ov_compute_weight_mat(
    input_size_node,
    output_size,
    scale_i,
    translation_i,
    kernel,
    antialias,
    ov_type,
):
    """Build the [m, n] resampling weight matrix.

    input_size_node: OV i32 scalar node — supports dynamic spatial dimensions.
    output_size: Python int — always statically known from the output_shape arg.
    """
    one = ov_opset.constant(1.0, dtype=ov_type).output(0)
    half = ov_opset.constant(0.5, dtype=ov_type).output(0)
    inv_scale = ov_opset.divide(one, scale_i).output(0)
    if antialias:
        kernel_scale = ov_opset.maximum(inv_scale, one).output(0)
    else:
        kernel_scale = one
    # sample_f = (arange(output_size) + 0.5) * inv_scale
    #            - translation_i * inv_scale - 0.5
    arange_n = ov_opset.constant(
        np.arange(output_size, dtype=np.float32), dtype=ov_type
    ).output(0)
    sample_f = ov_opset.subtract(
        ov_opset.subtract(
            ov_opset.multiply(ov_opset.add(arange_n, half), inv_scale),
            ov_opset.multiply(translation_i, inv_scale),
        ),
        half,
    ).output(0)
    input_size_f = ov_opset.convert(input_size_node, ov_type).output(0)
    arange_m = ov_opset.range(
        ov_opset.constant(0.0, dtype=ov_type).output(0),
        input_size_f,
        ov_opset.constant(1.0, dtype=ov_type).output(0),
        output_type=ov_type,
    ).output(0)
    sample_f_2d = ov_opset.unsqueeze(sample_f, axes=[0]).output(0)  # [1, n]
    arange_m_2d = ov_opset.unsqueeze(arange_m, axes=[1]).output(0)  # [m, 1]
    x = ov_opset.divide(
        ov_opset.abs(ov_opset.subtract(sample_f_2d, arange_m_2d)),
        kernel_scale,
    ).output(0)  # [m, n]
    weights = kernel(x)  # [m, n]
    axes_0_const = ov_opset.constant([0], Type.i32).output(0)
    total_weight_sum = ov_opset.reduce_sum(
        weights, axes_0_const, keep_dims=True
    ).output(0)  # [1, n]
    eps_val = 1000.0 * float(np.finfo(np.float32).eps)
    safe_denom = ov_opset.select(
        ov_opset.equal(total_weight_sum, ov_opset.constant(0.0, dtype=ov_type)),
        one,
        total_weight_sum,
    ).output(0)
    weights = ov_opset.select(
        ov_opset.greater(
            ov_opset.abs(total_weight_sum),
            ov_opset.constant(eps_val, dtype=ov_type),
        ),
        ov_opset.divide(weights, safe_denom),
        ov_opset.constant(0.0, dtype=ov_type),
    ).output(0)
    # Mask out-of-bounds sample positions
    upper_bound = ov_opset.subtract(
        input_size_f, ov_opset.constant(0.5, dtype=ov_type)
    ).output(0)
    in_bounds = ov_opset.logical_and(
        ov_opset.greater_equal(
            sample_f, ov_opset.constant(-0.5, dtype=ov_type)
        ),
        ov_opset.less_equal(sample_f, upper_bound),
    ).output(0)  # [n]
    in_bounds_2d = ov_opset.unsqueeze(in_bounds, axes=[0]).output(0)  # [1, n]
    return ov_opset.select(
        in_bounds_2d, weights, ov_opset.constant(0.0, dtype=ov_type)
    ).output(0)  # [m, n]


def _ov_scale_and_translate(
    x, output_shape, spatial_dims, scale_ov, translation_ov, kernel, antialias
):
    input_shape = x.get_partial_shape()
    ndim = input_shape.rank.get_length()
    x_ov_type = x.get_element_type()
    weight_ov_type = scale_ov.get_element_type()

    if len(spatial_dims) == 0:
        return OpenVINOKerasTensor(x)

    use_rounding = x_ov_type.is_integral()
    if use_rounding:
        output = ov_opset.convert(x, Type.f32).output(0)
    else:
        output = x

    axis_0 = ov_opset.constant(0, dtype=Type.i32).output(0)
    for i, d in enumerate(spatial_dims):
        d = d % ndim
        output_shape_node = ov_opset.shape_of(
            output, output_type=Type.i32
        ).output(0)
        m_node = ov_opset.gather(
            output_shape_node,
            ov_opset.constant(d, dtype=Type.i32).output(0),
            axis_0,
        ).output(0)
        n = output_shape[d]

        idx_i = ov_opset.constant(i, dtype=Type.i32).output(0)
        scale_i = ov_opset.gather(scale_ov, idx_i, axis_0).output(0)
        translation_i = ov_opset.gather(translation_ov, idx_i, axis_0).output(0)

        w = _ov_compute_weight_mat(
            m_node, n, scale_i, translation_i, kernel, antialias, weight_ov_type
        )
        w = ov_opset.convert(w, output.get_element_type()).output(0)

        # tensordot(output, w, axes=((d,), (0,))) then moveaxis(-1, d):
        # move axis d to last, matmul with w, move result back to d
        perm = list(range(ndim))
        perm.remove(d)
        perm.append(d)
        perm_const = ov_opset.constant(perm, dtype=Type.i32).output(0)
        output = ov_opset.transpose(output, perm_const).output(0)
        output = ov_opset.matmul(output, w, False, False).output(0)
        inv_perm = list(range(d)) + [ndim - 1] + list(range(d, ndim - 1))
        inv_perm_const = ov_opset.constant(inv_perm, dtype=Type.i32).output(0)
        output = ov_opset.transpose(output, inv_perm_const).output(0)

    if use_rounding:
        dtype_min = float(DTYPES_MIN[x_ov_type])
        dtype_max = float(DTYPES_MAX[x_ov_type])
        output = ov_opset.round(output, mode="half_to_even").output(0)
        output = ov_opset.clamp(output, dtype_min, dtype_max).output(0)
        output = ov_opset.convert(output, x_ov_type).output(0)

    return OpenVINOKerasTensor(output)


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
    dtype = backend.result_type(scale.dtype, translation.dtype)
    scale = cast(scale, dtype)
    translation = cast(translation, dtype)
    kernel = _ov_kernels[method]
    return _ov_scale_and_translate(
        get_ov_output(images),
        output_shape,
        spatial_dims,
        get_ov_output(scale),
        get_ov_output(translation),
        kernel,
        antialias,
    )
