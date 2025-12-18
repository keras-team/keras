import openvino.opset14 as ov_opset
from keras.src import backend
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import get_ov_output


def rgb_to_grayscale(images, data_format=None):
    images = get_ov_output(images)
    data_format = backend.standardize_data_format(data_format)

    # Weights for luminosity conversion: 0.2989 * R + 0.5870 * G + 0.1140 * B
    weights_list = [0.2989, 0.5870, 0.1140]
    dtype = images.get_element_type()

    rank = images.get_partial_shape().rank.get_length()
    if data_format == "channels_last":
        channel_axis = -1
    else:  # channels_first
        channel_axis = 1 if rank == 4 else 0

    # Reshape weights for broadcasting
    weights_shape = [1] * rank
    # handle negative axis
    axis_index = channel_axis if channel_axis >= 0 else rank + channel_axis
    weights_shape[axis_index] = 3

    weights = ov_opset.constant(weights_list, dtype)
    weights_node = ov_opset.reshape(
        weights, ov_opset.constant(weights_shape, "i64")
    )

    # Multiply and reduce sum is equivalent to a dot product.
    weighted_images = ov_opset.multiply(images, weights_node.output(0))
    gray = ov_opset.reduce_sum(
        weighted_images,
        axis=ov_opset.constant(channel_axis, "i32"),
        keep_dims=True,
    )

    return OpenVINOKerasTensor(gray.output(0))


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
    raise NotImplementedError("`resize` is not supported with openvino backend")


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
