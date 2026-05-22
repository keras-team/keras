import itertools

import numpy as np
import openvino.opset15 as ov_opset
from openvino import Type

from keras.src import backend
from keras.src.backend.openvino.core import DTYPES_MAX
from keras.src.backend.openvino.core import DTYPES_MIN
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import cast
from keras.src.backend.openvino.core import convert_to_numpy
from keras.src.backend.openvino.core import convert_to_tensor
from keras.src.backend.openvino.core import get_ov_output
from keras.src.backend.openvino.random import _random_normal
from keras.src.random.seed_generator import draw_seed

AFFINE_TRANSFORM_INTERPOLATIONS = {"nearest": 0, "bilinear": 1}
AFFINE_TRANSFORM_FILL_MODES = {
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
RESIZE_INTERPOLATIONS = {
    "bilinear": "linear",
    "nearest": "nearest",
    "bicubic": "cubic",
}
# Lanczos methods are handled via _ov_scale_and_translate
# instead of ov_opset.interpolate, which does not support them natively.
LANCZOS_INTERPOLATIONS = {
    "lanczos3": "lanczos3",
    "lanczos5": "lanczos5",
}
ALL_RESIZE_INTERPOLATIONS = tuple(RESIZE_INTERPOLATIONS) + tuple(
    LANCZOS_INTERPOLATIONS
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
    if interpolation not in ALL_RESIZE_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{ALL_RESIZE_INTERPOLATIONS}. Received: "
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

    if interpolation in LANCZOS_INTERPOLATIONS:
        # routed through the weight-matrix resampler used by
        # scale_and_translate, which has lanczos kernel support.
        post_crop_shape = ov_opset.shape_of(images, Type.i32).output(0)
        src_h = ov_opset.convert(
            _gather_dim(post_crop_shape, height_axis_index), Type.f32
        ).output(0)
        src_w = ov_opset.convert(
            _gather_dim(post_crop_shape, width_axis_index), Type.f32
        ).output(0)
        scale_h = ov_opset.divide(
            ov_opset.constant(float(target_height), Type.f32).output(0),
            src_h,
        ).output(0)
        scale_w = ov_opset.divide(
            ov_opset.constant(float(target_width), Type.f32).output(0),
            src_w,
        ).output(0)
        scale_ov = ov_opset.concat([scale_h, scale_w], axis=0).output(0)
        translation_ov = ov_opset.constant([0.0, 0.0], dtype=Type.f32).output(0)
        output_shape_map = {
            height_axis_index: target_height,
            width_axis_index: target_width,
        }
        spatial_dims = [height_axis_index, width_axis_index]
        kernel = _ov_kernels[LANCZOS_INTERPOLATIONS[interpolation]]
        return _ov_scale_and_translate(
            images,
            output_shape_map,
            spatial_dims,
            scale_ov,
            translation_ov,
            kernel,
            antialias,
        )

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


def _ov_build_affine_coords(images_ov, transform_ov):
    """Build transformed coordinates for affine_transform using OV opsets.

    images_ov: [B, H, W, C] f32
    transform_ov: [B, 8] f32 (TF-convention flat transform)
    Returns coords: [4, B, H, W, C] f32 — (batch, row, col, chan) per pixel.
    """
    shape_node = ov_opset.shape_of(images_ov, output_type=Type.i32).output(0)
    axis0 = ov_opset.constant(0, Type.i32).output(0)

    def dim(i):
        return ov_opset.gather(
            shape_node, ov_opset.constant(i, Type.i32).output(0), axis0
        ).output(0)

    B = dim(0)
    H = dim(1)
    W = dim(2)
    C = dim(3)

    H_f = ov_opset.convert(H, Type.f32).output(0)
    W_f = ov_opset.convert(W, Type.f32).output(0)
    C_f = ov_opset.convert(C, Type.f32).output(0)
    zero_f = ov_opset.constant(0.0, Type.f32).output(0)
    one_f = ov_opset.constant(1.0, Type.f32).output(0)
    r_h = ov_opset.range(zero_f, H_f, one_f, output_type=Type.f32).output(0)
    r_w = ov_opset.range(zero_f, W_f, one_f, output_type=Type.f32).output(0)
    r_c = ov_opset.range(zero_f, C_f, one_f, output_type=Type.f32).output(0)

    def s1d(scalar):
        return ov_opset.reshape(
            scalar, ov_opset.constant([1], Type.i32).output(0), False
        ).output(0)

    hwc_shape = ov_opset.concat([s1d(H), s1d(W), s1d(C)], axis=0).output(0)
    grid_h = ov_opset.broadcast(
        ov_opset.reshape(
            r_h,
            ov_opset.concat(
                [s1d(H), ov_opset.constant([1, 1], Type.i32).output(0)], axis=0
            ).output(0),
            False,
        ).output(0),
        hwc_shape,
    ).output(0)
    grid_w = ov_opset.broadcast(
        ov_opset.reshape(
            r_w,
            ov_opset.concat(
                [
                    ov_opset.constant([1], Type.i32).output(0),
                    s1d(W),
                    ov_opset.constant([1], Type.i32).output(0),
                ],
                axis=0,
            ).output(0),
            False,
        ).output(0),
        hwc_shape,
    ).output(0)
    grid_c = ov_opset.broadcast(
        ov_opset.reshape(
            r_c,
            ov_opset.concat(
                [ov_opset.constant([1, 1], Type.i32).output(0), s1d(C)], axis=0
            ).output(0),
            False,
        ).output(0),
        hwc_shape,
    ).output(0)

    neg1 = ov_opset.constant([-1], Type.i32).output(0)
    flat_h = ov_opset.reshape(grid_h, neg1, False).output(0)
    flat_w = ov_opset.reshape(grid_w, neg1, False).output(0)
    one_vec = ov_opset.broadcast(
        ov_opset.constant(1.0, Type.f32).output(0),
        ov_opset.shape_of(flat_h, output_type=Type.i32).output(0),
    ).output(0)
    pts = ov_opset.concat(
        [
            ov_opset.unsqueeze(flat_h, axes=[1]).output(0),
            ov_opset.unsqueeze(flat_w, axes=[1]).output(0),
            ov_opset.unsqueeze(one_vec, axes=[1]).output(0),
        ],
        axis=1,
    ).output(0)

    def t_col(i):
        return ov_opset.gather(
            transform_ov,
            ov_opset.constant(i, Type.i32).output(0),
            ov_opset.constant(1, Type.i32).output(0),
        ).output(0)  # [B]

    a0, a1, a2 = t_col(0), t_col(1), t_col(2)
    b0, b1, b2 = t_col(3), t_col(4), t_col(5)
    ones_b = ov_opset.broadcast(
        ov_opset.constant(1.0, Type.f32).output(0),
        ov_opset.shape_of(a0, output_type=Type.i32).output(0),
    ).output(0)
    zeros_b = ov_opset.broadcast(
        ov_opset.constant(0.0, Type.f32).output(0),
        ov_opset.shape_of(a0, output_type=Type.i32).output(0),
    ).output(0)

    # Numpy reference matrix (after swap + zero-out offset col):
    #   T = [[b1, a1, 0], [b0, a0, 0], [0, 0, 1]]   shape [3,3]
    # offset = [b2, a2, 0]
    # coords = pts @ T + offset  (right-multiply)
    def make_col(c0, c1, c2):
        col = ov_opset.concat(
            [
                ov_opset.unsqueeze(c0, axes=[1]).output(0),
                ov_opset.unsqueeze(c1, axes=[1]).output(0),
                ov_opset.unsqueeze(c2, axes=[1]).output(0),
            ],
            axis=1,
        ).output(0)  # [B, 3]
        return ov_opset.unsqueeze(col, axes=[2]).output(0)  # [B, 3, 1]

    # T as columns: col0=[b1,b0,0], col1=[a1,a0,0], col2=[0,0,1]
    col0 = make_col(b1, b0, zeros_b)  # [B, 3, 1]
    col1 = make_col(a1, a0, zeros_b)  # [B, 3, 1]
    col2 = make_col(zeros_b, zeros_b, ones_b)  # [B, 3, 1]
    mat = ov_opset.concat([col0, col1, col2], axis=2).output(0)  # [B, 3, 3]

    offset_row = ov_opset.unsqueeze(b2, axes=[1]).output(0)  # [B, 1]
    offset_col = ov_opset.unsqueeze(a2, axes=[1]).output(0)  # [B, 1]
    offset_chan = ov_opset.broadcast(
        ov_opset.constant(0.0, Type.f32).output(0),
        ov_opset.reshape(
            B, ov_opset.constant([1], Type.i32).output(0), False
        ).output(0),
    ).output(0)
    offset_chan = ov_opset.unsqueeze(offset_chan, axes=[1]).output(0)  # [B, 1]
    offset_3 = ov_opset.concat(
        [offset_row, offset_col, offset_chan], axis=1
    ).output(0)  # [B, 3]

    pts_b = ov_opset.unsqueeze(pts, axes=[0]).output(0)  # [1, N, 3]
    pts_shape = ov_opset.shape_of(pts, output_type=Type.i32).output(0)  # [2]
    B_N_3 = ov_opset.concat([s1d(B), pts_shape], axis=0).output(0)  # [B, N, 3]
    pts_b = ov_opset.broadcast(pts_b, B_N_3).output(0)  # [B, N, 3]

    transformed = ov_opset.matmul(pts_b, mat, False, False).output(
        0
    )  # [B, N, 3]

    offset_b = ov_opset.unsqueeze(offset_3, axes=[1]).output(0)  # [B, 1, 3]
    transformed = ov_opset.add(transformed, offset_b).output(0)  # [B, N, 3]

    transformed_t = ov_opset.transpose(
        transformed, ov_opset.constant([0, 2, 1], Type.i32).output(0)
    ).output(0)  # [B, 3, N]
    coord_row = ov_opset.squeeze(
        ov_opset.gather(
            transformed_t,
            ov_opset.constant([0], Type.i32).output(0),
            ov_opset.constant(1, Type.i32).output(0),
        ).output(0),  # [B, 1, N]
        axes=[1],
    ).output(0)  # [B, N]
    coord_col = ov_opset.squeeze(
        ov_opset.gather(
            transformed_t,
            ov_opset.constant([1], Type.i32).output(0),
            ov_opset.constant(1, Type.i32).output(0),
        ).output(0),  # [B, 1, N]
        axes=[1],
    ).output(0)  # [B, N]
    bhwc_shape = ov_opset.concat(
        [s1d(B), s1d(H), s1d(W), s1d(C)], axis=0
    ).output(0)
    coord_row = ov_opset.reshape(coord_row, bhwc_shape, False).output(0)
    coord_col = ov_opset.reshape(coord_col, bhwc_shape, False).output(0)
    # chan coords are the same for all batch items
    chan_bhwc = ov_opset.broadcast(
        ov_opset.reshape(
            ov_opset.reshape(grid_c, neg1, False).output(0),
            ov_opset.concat(
                [
                    ov_opset.constant([1], Type.i32).output(0),
                    s1d(H),
                    s1d(W),
                    s1d(C),
                ],
                axis=0,
            ).output(0),
            False,
        ).output(0),
        bhwc_shape,
    ).output(0)
    zero_f = ov_opset.constant(0.0, Type.f32).output(0)
    one_f = ov_opset.constant(1.0, Type.f32).output(0)
    B_f = ov_opset.convert(B, Type.f32).output(0)
    r_b = ov_opset.range(zero_f, B_f, one_f, output_type=Type.f32).output(0)
    batch_bhwc = ov_opset.broadcast(
        ov_opset.reshape(
            r_b,
            ov_opset.concat(
                [s1d(B), ov_opset.constant([1, 1, 1], Type.i32).output(0)],
                axis=0,
            ).output(0),
            False,
        ).output(0),
        bhwc_shape,
    ).output(0)

    # Stack to [4, B, H, W, C] — (batch, row, col, chan)
    coords = ov_opset.concat(
        [
            ov_opset.unsqueeze(batch_bhwc, axes=[0]).output(0),
            ov_opset.unsqueeze(coord_row, axes=[0]).output(0),
            ov_opset.unsqueeze(coord_col, axes=[0]).output(0),
            ov_opset.unsqueeze(chan_bhwc, axes=[0]).output(0),
        ],
        axis=0,
    ).output(0)
    return coords


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
    images_ov = get_ov_output(images)
    transform_ov = get_ov_output(transform)

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

    ov_type = images_ov.get_element_type()
    compute_type = Type.f32

    need_squeeze = False
    if len(images.shape) == 3:
        images_ov = ov_opset.unsqueeze(images_ov, axes=[0]).output(0)
        need_squeeze = True
    if len(transform.shape) == 1:
        transform_ov = ov_opset.unsqueeze(transform_ov, axes=[0]).output(0)

    if data_format == "channels_first":
        images_ov = ov_opset.transpose(
            images_ov,
            ov_opset.constant([0, 2, 3, 1], Type.i32).output(0),
        ).output(0)

    images_ov = ov_opset.convert(images_ov, compute_type).output(0)
    transform_ov = ov_opset.convert(transform_ov, compute_type).output(0)

    coords = _ov_build_affine_coords(images_ov, transform_ov)
    affined = map_coordinates(
        OpenVINOKerasTensor(images_ov),
        OpenVINOKerasTensor(coords),
        order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
        fill_mode=fill_mode,
        fill_value=fill_value,
    )
    affined = get_ov_output(affined)

    if ov_type.is_integral():
        affined = ov_opset.round(affined, mode="half_to_even").output(0)
    affined = ov_opset.convert(affined, ov_type).output(0)

    if data_format == "channels_first":
        affined = ov_opset.transpose(
            affined,
            ov_opset.constant([0, 3, 1, 2], Type.i32).output(0),
        ).output(0)
    if need_squeeze:
        affined = ov_opset.squeeze(affined, axes=[0]).output(0)

    return OpenVINOKerasTensor(affined)


def compute_homography_matrix(start_points, end_points):
    start_points = convert_to_tensor(start_points, dtype="float32")
    end_points = convert_to_tensor(end_points, dtype="float32")
    sp_ov = get_ov_output(start_points)
    ep_ov = get_ov_output(end_points)

    axis0 = ov_opset.constant(0, Type.i32).output(0)
    axis1 = ov_opset.constant(1, Type.i32).output(0)

    def _split_points(pts_ov):
        corners = [
            ov_opset.squeeze(
                ov_opset.gather(
                    pts_ov,
                    ov_opset.constant([c], Type.i32).output(0),
                    axis1,
                ).output(0),
                axes=[1],
            ).output(0)
            for c in range(4)
        ]
        result = []
        for pt in corners:
            x = ov_opset.squeeze(
                ov_opset.gather(
                    pt, ov_opset.constant([0], Type.i32).output(0), axis1
                ).output(0),
                axes=[1],
            ).output(0)
            y = ov_opset.squeeze(
                ov_opset.gather(
                    pt, ov_opset.constant([1], Type.i32).output(0), axis1
                ).output(0),
                axes=[1],
            ).output(0)
            result.append((x, y))
        return result

    end_pts = _split_points(ep_ov)
    start_pts = _split_points(sp_ov)

    B_shape = ov_opset.shape_of(sp_ov, output_type=Type.i32).output(0)
    B = ov_opset.gather(
        B_shape, ov_opset.constant(0, Type.i32).output(0), axis0
    ).output(0)

    def ones():
        return ov_opset.broadcast(
            ov_opset.constant(1.0, Type.f32).output(0),
            ov_opset.reshape(
                B, ov_opset.constant([1], Type.i32).output(0), False
            ).output(0),
        ).output(0)

    def zeros():
        return ov_opset.broadcast(
            ov_opset.constant(0.0, Type.f32).output(0),
            ov_opset.reshape(
                B, ov_opset.constant([1], Type.i32).output(0), False
            ).output(0),
        ).output(0)

    def neg(x):
        return ov_opset.multiply(
            x, ov_opset.constant(-1.0, Type.f32).output(0)
        ).output(0)

    def mul(a, b):
        return ov_opset.multiply(a, b).output(0)

    def make_two_rows(ex, ey, sx, sy):
        o = ones()
        z = zeros()

        def row_from_elems(elems):
            return ov_opset.concat(
                [ov_opset.unsqueeze(e, axes=[1]).output(0) for e in elems],
                axis=1,
            ).output(0)

        row1 = row_from_elems(
            [ex, ey, o, z, z, z, neg(mul(sx, ex)), neg(mul(sx, ey))]
        )
        row2 = row_from_elems(
            [z, z, z, ex, ey, o, neg(mul(sy, ex)), neg(mul(sy, ey))]
        )
        return row1, row2

    rows = []
    rhs_cols = []
    for corner in range(4):
        ex, ey = end_pts[corner]
        sx, sy = start_pts[corner]
        r1, r2 = make_two_rows(ex, ey, sx, sy)
        rows.extend([r1, r2])
        rhs_cols.extend([sx, sy])

    coeff_mat = ov_opset.concat(
        [ov_opset.unsqueeze(r, axes=[1]).output(0) for r in rows], axis=1
    ).output(0)  # [B, 8, 8]
    rhs_2d = ov_opset.concat(
        [ov_opset.unsqueeze(c, axes=[1]).output(0) for c in rhs_cols], axis=1
    ).output(0)  # [B, 8]
    rhs = ov_opset.unsqueeze(rhs_2d, axes=[2]).output(0)  # [B, 8, 1]

    coeff_inv = ov_opset.inverse(coeff_mat, adjoint=False).output(0)
    h = ov_opset.matmul(coeff_inv, rhs, False, False).output(0)  # [B, 8, 1]
    h = ov_opset.squeeze(h, axes=[2]).output(0)  # [B, 8]
    return h


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )

    images = convert_to_tensor(images)
    start_points = convert_to_tensor(start_points)
    end_points = convert_to_tensor(end_points)
    images_ov = get_ov_output(images)
    sp_ov = get_ov_output(start_points)
    ep_ov = get_ov_output(end_points)

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if len(start_points.shape) not in (2, 3) or tuple(start_points.shape)[
        -2:
    ] != (4, 2):
        raise ValueError(
            "Invalid start_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {start_points.shape}"
        )
    if len(end_points.shape) not in (2, 3) or tuple(end_points.shape)[-2:] != (
        4,
        2,
    ):
        raise ValueError(
            "Invalid end_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {end_points.shape}"
        )

    ov_type = images_ov.get_element_type()
    compute_type = Type.f32

    need_squeeze = False
    if len(images.shape) == 3:
        images_ov = ov_opset.unsqueeze(images_ov, axes=[0]).output(0)
        need_squeeze = True
    if len(start_points.shape) == 2:
        sp_ov = ov_opset.unsqueeze(sp_ov, axes=[0]).output(0)
    if len(end_points.shape) == 2:
        ep_ov = ov_opset.unsqueeze(ep_ov, axes=[0]).output(0)

    if data_format == "channels_first":
        images_ov = ov_opset.transpose(
            images_ov,
            ov_opset.constant([0, 2, 3, 1], Type.i32).output(0),
        ).output(0)

    images_ov = ov_opset.convert(images_ov, compute_type).output(0)

    transforms = get_ov_output(
        compute_homography_matrix(start_points, end_points)
    )

    shape_node = ov_opset.shape_of(images_ov, output_type=Type.i32).output(0)
    axis0 = ov_opset.constant(0, Type.i32).output(0)

    def dim(i):
        return ov_opset.gather(
            shape_node, ov_opset.constant(i, Type.i32).output(0), axis0
        ).output(0)

    B = dim(0)
    H = dim(1)
    W = dim(2)
    C = dim(3)

    H_f = ov_opset.convert(H, Type.f32).output(0)
    W_f = ov_opset.convert(W, Type.f32).output(0)
    zero_f = ov_opset.constant(0.0, Type.f32).output(0)
    one_f = ov_opset.constant(1.0, Type.f32).output(0)
    r_h = ov_opset.range(zero_f, H_f, one_f, output_type=Type.f32).output(
        0
    )  # [H]
    r_w = ov_opset.range(zero_f, W_f, one_f, output_type=Type.f32).output(
        0
    )  # [W]

    def p1d(scalar):
        return ov_opset.reshape(
            scalar, ov_opset.constant([1], Type.i32).output(0), False
        ).output(0)

    hw_shape = ov_opset.concat([p1d(H), p1d(W)], axis=0).output(0)
    # y: rows  [H, W]
    y = ov_opset.broadcast(
        ov_opset.reshape(
            r_h,
            ov_opset.concat(
                [p1d(H), ov_opset.constant([1], Type.i32).output(0)], axis=0
            ).output(0),
            False,
        ).output(0),
        hw_shape,
    ).output(0)
    # x: cols  [H, W]
    x = ov_opset.broadcast(
        ov_opset.reshape(
            r_w,
            ov_opset.concat(
                [ov_opset.constant([1], Type.i32).output(0), p1d(W)], axis=0
            ).output(0),
            False,
        ).output(0),
        hw_shape,
    ).output(0)

    neg1 = ov_opset.constant([-1], Type.i32).output(0)
    x_flat = ov_opset.reshape(x, neg1, False).output(0)  # [N]
    y_flat = ov_opset.reshape(y, neg1, False).output(0)  # [N]

    axis1 = ov_opset.constant(1, Type.i32).output(0)

    def h_col(i):
        return ov_opset.squeeze(
            ov_opset.gather(
                transforms, ov_opset.constant([i], Type.i32).output(0), axis1
            ).output(0),  # [B, 1]
            axes=[1],
        ).output(0)  # [B]

    a0, a1, a2 = h_col(0), h_col(1), h_col(2)
    a3, a4, a5 = h_col(3), h_col(4), h_col(5)
    a6, a7 = h_col(6), h_col(7)

    N_shape = ov_opset.shape_of(x_flat, output_type=Type.i32).output(0)
    BN_shape = ov_opset.concat(
        [
            ov_opset.reshape(
                B, ov_opset.constant([1], Type.i32).output(0), False
            ).output(0),
            N_shape,
        ],
        axis=0,
    ).output(0)

    def bcast(v):
        return ov_opset.broadcast(
            ov_opset.unsqueeze(v, axes=[1]).output(0), BN_shape
        ).output(0)

    x_bn = ov_opset.broadcast(
        ov_opset.unsqueeze(x_flat, axes=[0]).output(0), BN_shape
    ).output(0)
    y_bn = ov_opset.broadcast(
        ov_opset.unsqueeze(y_flat, axes=[0]).output(0), BN_shape
    ).output(0)

    # denom = a6*x + a7*y + 1
    one_bn = ov_opset.broadcast(
        ov_opset.constant(1.0, Type.f32).output(0), BN_shape
    ).output(0)
    denom = ov_opset.add(
        ov_opset.add(
            ov_opset.multiply(bcast(a6), x_bn).output(0),
            ov_opset.multiply(bcast(a7), y_bn).output(0),
        ).output(0),
        one_bn,
    ).output(0)

    # x_in = (a0*x + a1*y + a2) / denom
    x_in = ov_opset.divide(
        ov_opset.add(
            ov_opset.add(
                ov_opset.multiply(bcast(a0), x_bn).output(0),
                ov_opset.multiply(bcast(a1), y_bn).output(0),
            ).output(0),
            bcast(a2),
        ).output(0),
        denom,
    ).output(0)

    # y_in = (a3*x + a4*y + a5) / denom
    y_in = ov_opset.divide(
        ov_opset.add(
            ov_opset.add(
                ov_opset.multiply(bcast(a3), x_bn).output(0),
                ov_opset.multiply(bcast(a4), y_bn).output(0),
            ).output(0),
            bcast(a5),
        ).output(0),
        denom,
    ).output(0)

    bhw_shape = ov_opset.concat([p1d(B), p1d(H), p1d(W)], axis=0).output(0)
    y_in = ov_opset.reshape(y_in, bhw_shape, False).output(0)
    x_in = ov_opset.reshape(x_in, bhw_shape, False).output(0)

    C_f = ov_opset.convert(C, Type.f32).output(0)
    r_c = ov_opset.range(zero_f, C_f, one_f, output_type=Type.f32).output(
        0
    )  # [C]

    bhwc_shape = ov_opset.concat(
        [p1d(B), p1d(H), p1d(W), p1d(C)], axis=0
    ).output(0)

    y_in_bhwc = ov_opset.broadcast(
        ov_opset.unsqueeze(y_in, axes=[3]).output(0), bhwc_shape
    ).output(0)
    x_in_bhwc = ov_opset.broadcast(
        ov_opset.unsqueeze(x_in, axes=[3]).output(0), bhwc_shape
    ).output(0)
    chan_bhwc = ov_opset.broadcast(
        ov_opset.reshape(
            r_c,
            ov_opset.concat(
                [ov_opset.constant([1, 1, 1], Type.i32).output(0), p1d(C)],
                axis=0,
            ).output(0),
            False,
        ).output(0),
        bhwc_shape,
    ).output(0)
    B_f = ov_opset.convert(B, Type.f32).output(0)
    r_b = ov_opset.range(zero_f, B_f, one_f, output_type=Type.f32).output(0)
    batch_bhwc = ov_opset.broadcast(
        ov_opset.reshape(
            r_b,
            ov_opset.concat(
                [p1d(B), ov_opset.constant([1, 1, 1], Type.i32).output(0)],
                axis=0,
            ).output(0),
            False,
        ).output(0),
        bhwc_shape,
    ).output(0)

    # coords: [4, B, H, W, C] — (batch, row=y, col=x, chan)
    coords = ov_opset.concat(
        [
            ov_opset.unsqueeze(batch_bhwc, axes=[0]).output(0),
            ov_opset.unsqueeze(y_in_bhwc, axes=[0]).output(0),
            ov_opset.unsqueeze(x_in_bhwc, axes=[0]).output(0),
            ov_opset.unsqueeze(chan_bhwc, axes=[0]).output(0),
        ],
        axis=0,
    ).output(0)

    result = map_coordinates(
        OpenVINOKerasTensor(images_ov),
        OpenVINOKerasTensor(coords),
        order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
        fill_mode="constant",
        fill_value=fill_value,
    )
    result = get_ov_output(result)

    if ov_type.is_integral():
        result = ov_opset.round(result, mode="half_to_even").output(0)
    result = ov_opset.convert(result, ov_type).output(0)

    if data_format == "channels_first":
        result = ov_opset.transpose(
            result,
            ov_opset.constant([0, 3, 1, 2], Type.i32).output(0),
        ).output(0)
    if need_squeeze:
        result = ov_opset.squeeze(result, axes=[0]).output(0)

    return OpenVINOKerasTensor(result)


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
    def _create_gaussian_kernel(kernel_size, sigma):
        # Always build the kernel in f32 for numerical stability and
        # compatibility (bfloat16 / f16 are not fully supported by all ops).
        def _get_gaussian_kernel1d(size, sigma):
            x = ov_opset.subtract(
                ov_opset.range(0, size, 1, output_type=Type.f32).output(0),
                ov_opset.constant((size - 1) / 2.0, Type.f32).output(0),
            ).output(0)

            sigma_const = ov_opset.constant(float(sigma), Type.f32).output(0)
            exponent = ov_opset.divide(x, sigma_const).output(0)
            exponent = ov_opset.multiply(exponent, exponent).output(0)
            exponent = ov_opset.multiply(
                exponent, ov_opset.constant(-0.5, Type.f32).output(0)
            ).output(0)
            kernel1d = ov_opset.exp(exponent).output(0)
            kernel1d_sum = ov_opset.reduce_sum(
                kernel1d, reduction_axes=0, keep_dims=False
            ).output(0)

            return ov_opset.divide(kernel1d, kernel1d_sum).output(0)

        def _get_gaussian_kernel2d(size, sigma):
            kernel1d_x = _get_gaussian_kernel1d(size[0], sigma[0])
            kernel1d_y = _get_gaussian_kernel1d(size[1], sigma[1])

            # kernel1d_x has kH elements -> row vector [1, kH]
            kernel1d_x = ov_opset.reshape(
                kernel1d_x,
                ov_opset.constant([1, int(size[0])], Type.i32).output(0),
                False,
            ).output(0)

            # kernel1d_y has kW elements -> column vector [kW, 1]
            kernel1d_y = ov_opset.reshape(
                kernel1d_y,
                ov_opset.constant([int(size[1]), 1], Type.i32).output(0),
                False,
            ).output(0)
            return ov_opset.multiply(kernel1d_y, kernel1d_x).output(0)

        return _get_gaussian_kernel2d(kernel_size, sigma)

    data_format = backend.standardize_data_format(data_format)
    images = convert_to_tensor(images)
    input_shape = images.shape
    ov_type = get_ov_output(images).get_element_type()

    if len(input_shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={input_shape}"
        )

    # bfloat16 is not supported by all OV ops used here; promote to f32.
    # f16 constants in range() would mismatch with the f32 arithmetic
    # constants in the kernel builder, so promote f16 to f32 as well.
    compute_type = Type.f32 if ov_type in (Type.bf16, Type.f16) else ov_type
    images = get_ov_output(images)
    if compute_type != ov_type:
        images = ov_opset.convert(images, compute_type).output(0)

    need_squeeze = False
    if len(input_shape) == 3:
        images = ov_opset.unsqueeze(images, axes=[0]).output(0)
        need_squeeze = True

    if data_format == "channels_last":
        images = ov_opset.transpose(
            images,
            ov_opset.constant([0, 3, 1, 2], Type.i32).output(0),
        ).output(0)

    num_channels = ov_opset.gather(
        ov_opset.shape_of(images, Type.i32).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant(0, Type.i32).output(0),
    ).output(0)
    # Kernel is always built in f32; convert to compute_type before conv.
    kernel = _create_gaussian_kernel(kernel_size, sigma)
    kernel = ov_opset.convert(kernel, compute_type).output(0)

    kernel = ov_opset.reshape(
        kernel,
        ov_opset.constant(
            [1, 1, kernel_size[0], kernel_size[1]], Type.i32
        ).output(0),
        False,
    ).output(0)

    target_shape = ov_opset.concat(
        [
            num_channels,
            ov_opset.constant(
                [1, 1, kernel_size[0], kernel_size[1]], Type.i32
            ).output(0),
        ],
        axis=0,
    ).output(0)

    kernel = ov_opset.broadcast(kernel, target_shape).output(0)

    blurred_images = ov_opset.group_convolution(
        images,
        kernel,
        [1, 1],
        [(kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2],
        [
            kernel_size[0] - 1 - (kernel_size[0] - 1) // 2,
            kernel_size[1] - 1 - (kernel_size[1] - 1) // 2,
        ],
        [1, 1],
    ).output(0)

    # Cast back to the original dtype if we promoted for computation.
    if compute_type != ov_type:
        blurred_images = ov_opset.convert(blurred_images, ov_type).output(0)

    if data_format == "channels_last":
        blurred_images = ov_opset.transpose(
            blurred_images,
            ov_opset.constant([0, 2, 3, 1], Type.i32).output(0),
        ).output(0)

    if need_squeeze:
        blurred_images = ov_opset.squeeze(blurred_images, axes=[0]).output(0)

    return OpenVINOKerasTensor(blurred_images)


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
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS:
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
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    images = convert_to_tensor(images)
    images_ov = get_ov_output(images)
    ov_type = images_ov.get_element_type()
    compute_type = Type.f32

    need_squeeze = False
    if len(images.shape) == 3:
        images_ov = ov_opset.unsqueeze(images_ov, axes=[0]).output(0)
        need_squeeze = True

    if data_format == "channels_last":
        images_ov_cf = ov_opset.transpose(
            images_ov,
            ov_opset.constant([0, 3, 1, 2], Type.i32).output(0),
        ).output(0)
    else:
        images_ov_cf = images_ov

    images_ov_cf = ov_opset.convert(images_ov_cf, compute_type).output(0)

    shape_node = ov_opset.shape_of(images_ov_cf, output_type=Type.i32).output(0)
    axis0 = ov_opset.constant(0, Type.i32).output(0)

    def dim(i):
        return ov_opset.gather(
            shape_node, ov_opset.constant(i, Type.i32).output(0), axis0
        ).output(0)

    B = dim(0)
    C = dim(1)
    H = dim(2)
    W = dim(3)

    sigma_val = float(sigma)
    alpha_val = float(alpha)
    kernel_size_1d = int(6 * sigma_val) | 1

    # OV random ops require static seed attributes, so symbolic seeds must be
    # materialized via convert_to_numpy. This is an unavoidable sync point given
    # the OV backend's stateless random design.
    seed_val = draw_seed(seed)
    if isinstance(seed_val, OpenVINOKerasTensor):
        s = convert_to_numpy(seed_val)
    else:
        s = seed_val.data
    seed1 = max(1, int(s[0]) & 0x7FFFFFFF)
    seed2 = max(1, int(s[1]) & 0x7FFFFFFF) if len(s) > 1 else 1

    def to_1d(scalar):
        return ov_opset.reshape(
            scalar, ov_opset.constant([1], Type.i32).output(0), False
        ).output(0)

    bhw_shape = ov_opset.concat(
        [to_1d(B), to_1d(H), to_1d(W)],
        axis=0,
    ).output(0)

    dx = _random_normal(bhw_shape, Type.f32, seed1, seed2)  # [B, H, W]
    dy = _random_normal(bhw_shape, Type.f32, seed1 + 1, seed2)  # [B, H, W]

    # Scale by sigma before gaussian blur
    sigma_const = ov_opset.constant(sigma_val, Type.f32).output(0)
    dx = ov_opset.multiply(dx, sigma_const).output(0)
    dy = ov_opset.multiply(dy, sigma_const).output(0)

    # Apply gaussian blur to smooth the displacement fields
    # Add channel dim: [B, 1, H, W] for channels_first gaussian_blur
    dx_4d = ov_opset.unsqueeze(dx, axes=[1]).output(0)
    dy_4d = ov_opset.unsqueeze(dy, axes=[1]).output(0)

    dx_blurred = gaussian_blur(
        OpenVINOKerasTensor(dx_4d),
        kernel_size=(kernel_size_1d, kernel_size_1d),
        sigma=(sigma_val, sigma_val),
        data_format="channels_first",
    )
    dy_blurred = gaussian_blur(
        OpenVINOKerasTensor(dy_4d),
        kernel_size=(kernel_size_1d, kernel_size_1d),
        sigma=(sigma_val, sigma_val),
        data_format="channels_first",
    )
    dx_blurred = ov_opset.squeeze(get_ov_output(dx_blurred), axes=[1]).output(
        0
    )  # [B, H, W]
    dy_blurred = ov_opset.squeeze(get_ov_output(dy_blurred), axes=[1]).output(
        0
    )  # [B, H, W]

    H_f = ov_opset.convert(H, Type.f32).output(0)
    W_f = ov_opset.convert(W, Type.f32).output(0)
    zero_f = ov_opset.constant(0.0, Type.f32).output(0)
    one_f = ov_opset.constant(1.0, Type.f32).output(0)
    r_h = ov_opset.range(zero_f, H_f, one_f, output_type=Type.f32).output(0)
    r_w = ov_opset.range(zero_f, W_f, one_f, output_type=Type.f32).output(0)
    hw_shape = ov_opset.concat([to_1d(H), to_1d(W)], axis=0).output(0)

    y_base = ov_opset.broadcast(
        ov_opset.reshape(
            r_h,
            ov_opset.concat(
                [to_1d(H), ov_opset.constant([1], Type.i32).output(0)], axis=0
            ).output(0),
            False,
        ).output(0),
        hw_shape,
    ).output(0)  # [H, W]
    x_base = ov_opset.broadcast(
        ov_opset.reshape(
            r_w,
            ov_opset.concat(
                [ov_opset.constant([1], Type.i32).output(0), to_1d(W)], axis=0
            ).output(0),
            False,
        ).output(0),
        hw_shape,
    ).output(0)  # [H, W]

    bhw_bcast = ov_opset.concat([to_1d(B), to_1d(H), to_1d(W)], axis=0).output(
        0
    )
    y_base_b = ov_opset.broadcast(
        ov_opset.unsqueeze(y_base, axes=[0]).output(0), bhw_bcast
    ).output(0)
    x_base_b = ov_opset.broadcast(
        ov_opset.unsqueeze(x_base, axes=[0]).output(0), bhw_bcast
    ).output(0)

    alpha_const = ov_opset.constant(alpha_val, Type.f32).output(0)
    distorted_x = ov_opset.add(
        x_base_b, ov_opset.multiply(alpha_const, dx_blurred).output(0)
    ).output(0)
    distorted_y = ov_opset.add(
        y_base_b, ov_opset.multiply(alpha_const, dy_blurred).output(0)
    ).output(0)

    # Build coords [3, B, H, W, C] — (row=y, col=x, chan) per output pixel
    C_f = ov_opset.convert(C, Type.f32).output(0)
    r_c = ov_opset.range(zero_f, C_f, one_f, output_type=Type.f32).output(0)

    bhwc_shape = ov_opset.concat(
        [to_1d(B), to_1d(H), to_1d(W), to_1d(C)], axis=0
    ).output(0)

    y_bhwc = ov_opset.broadcast(
        ov_opset.unsqueeze(distorted_y, axes=[3]).output(0), bhwc_shape
    ).output(0)
    x_bhwc = ov_opset.broadcast(
        ov_opset.unsqueeze(distorted_x, axes=[3]).output(0), bhwc_shape
    ).output(0)
    chan_bhwc = ov_opset.broadcast(
        ov_opset.reshape(
            r_c,
            ov_opset.concat(
                [ov_opset.constant([1, 1, 1], Type.i32).output(0), to_1d(C)],
                axis=0,
            ).output(0),
            False,
        ).output(0),
        bhwc_shape,
    ).output(0)

    B_f = ov_opset.convert(B, Type.f32).output(0)
    r_b = ov_opset.range(zero_f, B_f, one_f, output_type=Type.f32).output(0)
    batch_bhwc = ov_opset.broadcast(
        ov_opset.reshape(
            r_b,
            ov_opset.concat(
                [to_1d(B), ov_opset.constant([1, 1, 1], Type.i32).output(0)],
                axis=0,
            ).output(0),
            False,
        ).output(0),
        bhwc_shape,
    ).output(0)

    # coords: [4, B, H, W, C] — (batch, row=y, col=x, chan)
    coords = ov_opset.concat(
        [
            ov_opset.unsqueeze(batch_bhwc, axes=[0]).output(0),
            ov_opset.unsqueeze(y_bhwc, axes=[0]).output(0),
            ov_opset.unsqueeze(x_bhwc, axes=[0]).output(0),
            ov_opset.unsqueeze(chan_bhwc, axes=[0]).output(0),
        ],
        axis=0,
    ).output(0)  # [4, B, H, W, C]

    # images_ov_cf is [B, C, H, W] but map_coordinates needs [B, H, W, C]
    images_bhwc = ov_opset.transpose(
        images_ov_cf,
        ov_opset.constant([0, 2, 3, 1], Type.i32).output(0),
    ).output(0)

    result = map_coordinates(
        OpenVINOKerasTensor(images_bhwc),
        OpenVINOKerasTensor(coords),
        order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
        fill_mode=fill_mode,
        fill_value=fill_value,
    )
    result = get_ov_output(result)

    if ov_type.is_integral():
        result = ov_opset.round(result, mode="half_to_even").output(0)
    result = ov_opset.convert(result, ov_type).output(0)

    if data_format == "channels_first":
        result = ov_opset.transpose(
            result,
            ov_opset.constant([0, 3, 1, 2], Type.i32).output(0),
        ).output(0)
    if need_squeeze:
        result = ov_opset.squeeze(result, axes=[0]).output(0)

    return OpenVINOKerasTensor(result)


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


def sobel_edges(images, data_format=None):
    raise NotImplementedError(
        "`sobel_edges` is not supported with openvino backend"
    )
