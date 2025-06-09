def rgb_to_grayscale(images, data_format=None):
    raise NotImplementedError(
        "`rgb_to_grayscale` is not supported with openvino backend"
    )


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


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0
):
    raise NotImplementedError(
        "`map_coordinates` is not supported with openvino backend"
    )
