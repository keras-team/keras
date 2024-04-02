"""Utilities related to image handling."""

import io
import pathlib
import warnings

import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export

try:
    from PIL import Image as pil_image

    try:
        pil_image_resampling = pil_image.Resampling
    except AttributeError:
        pil_image_resampling = pil_image
except ImportError:
    pil_image = None
    pil_image_resampling = None


if pil_image_resampling is not None:
    PIL_INTERPOLATION_METHODS = {
        "nearest": pil_image_resampling.NEAREST,
        "bilinear": pil_image_resampling.BILINEAR,
        "bicubic": pil_image_resampling.BICUBIC,
        "hamming": pil_image_resampling.HAMMING,
        "box": pil_image_resampling.BOX,
        "lanczos": pil_image_resampling.LANCZOS,
    }


@keras_export(
    [
        "keras.utils.array_to_img",
        "keras.preprocessing.image.array_to_img",
    ]
)
def array_to_img(x, data_format=None, scale=True, dtype=None):
    """Converts a 3D NumPy array to a PIL Image instance.

    Example:

    ```python
    from PIL import Image
    img = np.random.random(size=(100, 100, 3))
    pil_img = keras.utils.array_to_img(img)
    ```

    Args:
        x: Input data, in any form that can be converted to a NumPy array.
        data_format: Image data format, can be either `"channels_first"` or
            `"channels_last"`. Defaults to `None`, in which case the global
            setting `keras.backend.image_data_format()` is used (unless you
            changed it, it defaults to `"channels_last"`).
        scale: Whether to rescale the image such that minimum and maximum values
            are 0 and 255 respectively. Defaults to `True`.
        dtype: Dtype to use. `None` means the global setting
            `keras.backend.floatx()` is used (unless you changed it, it
            defaults to `"float32"`). Defaults to `None`.

    Returns:
        A PIL Image instance.
    """

    data_format = backend.standardize_data_format(data_format)
    if dtype is None:
        dtype = backend.floatx()
    if pil_image is None:
        raise ImportError(
            "Could not import PIL.Image. "
            "The use of `array_to_img` requires PIL."
        )
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError(
            "Expected image array to have rank 3 (single image). "
            f"Got array with shape: {x.shape}"
        )

    # Original NumPy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == "channels_first":
        x = x.transpose(1, 2, 0)
    if scale:
        x = x - np.min(x)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype("uint8"), "RGBA")
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype("uint8"), "RGB")
    elif x.shape[2] == 1:
        # grayscale
        if np.max(x) > 255:
            # 32-bit signed integer grayscale image. PIL mode "I"
            return pil_image.fromarray(x[:, :, 0].astype("int32"), "I")
        return pil_image.fromarray(x[:, :, 0].astype("uint8"), "L")
    else:
        raise ValueError(f"Unsupported channel number: {x.shape[2]}")


@keras_export(
    [
        "keras.utils.img_to_array",
        "keras.preprocessing.image.img_to_array",
    ]
)
def img_to_array(img, data_format=None, dtype=None):
    """Converts a PIL Image instance to a NumPy array.

    Example:

    ```python
    from PIL import Image
    img_data = np.random.random(size=(100, 100, 3))
    img = keras.utils.array_to_img(img_data)
    array = keras.utils.image.img_to_array(img)
    ```

    Args:
        img: Input PIL Image instance.
        data_format: Image data format, can be either `"channels_first"` or
            `"channels_last"`. Defaults to `None`, in which case the global
            setting `keras.backend.image_data_format()` is used (unless you
            changed it, it defaults to `"channels_last"`).
        dtype: Dtype to use. `None` means the global setting
            `keras.backend.floatx()` is used (unless you changed it, it
            defaults to `"float32"`).

    Returns:
        A 3D NumPy array.
    """

    data_format = backend.standardize_data_format(data_format)
    if dtype is None:
        dtype = backend.floatx()
    # NumPy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == "channels_first":
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == "channels_first":
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError(f"Unsupported image shape: {x.shape}")
    return x


@keras_export(["keras.utils.save_img", "keras.preprocessing.image.save_img"])
def save_img(path, x, data_format=None, file_format=None, scale=True, **kwargs):
    """Saves an image stored as a NumPy array to a path or file object.

    Args:
        path: Path or file object.
        x: NumPy array.
        data_format: Image data format, either `"channels_first"` or
            `"channels_last"`.
        file_format: Optional file format override. If omitted, the format to
            use is determined from the filename extension. If a file object was
            used instead of a filename, this parameter should always be used.
        scale: Whether to rescale image values to be within `[0, 255]`.
        **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
    """
    data_format = backend.standardize_data_format(data_format)
    img = array_to_img(x, data_format=data_format, scale=scale)
    if img.mode == "RGBA" and (file_format == "jpg" or file_format == "jpeg"):
        warnings.warn(
            "The JPG format does not support RGBA images, converting to RGB."
        )
        img = img.convert("RGB")
    img.save(path, format=file_format, **kwargs)


@keras_export(["keras.utils.load_img", "keras.preprocessing.image.load_img"])
def load_img(
    path,
    color_mode="rgb",
    target_size=None,
    interpolation="nearest",
    keep_aspect_ratio=False,
):
    """Loads an image into PIL format.

    Example:

    ```python
    image = keras.utils.load_img(image_path)
    input_arr = keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    ```

    Args:
        path: Path to image file.
        color_mode: One of `"grayscale"`, `"rgb"`, `"rgba"`. Default: `"rgb"`.
            The desired image format.
        target_size: Either `None` (default to original size) or tuple of ints
            `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image. Supported
            methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
            If PIL version 1.1.3 or newer is installed, `"lanczos"`
            is also supported. If PIL version 3.4.0 or newer is installed,
            `"box"` and `"hamming"` are also
            supported. By default, `"nearest"` is used.
        keep_aspect_ratio: Boolean, whether to resize images to a target
            size without aspect ratio distortion. The image is cropped in
            the center with target aspect ratio before resizing.

    Returns:
        A PIL Image instance.
    """
    if pil_image is None:
        raise ImportError(
            "Could not import PIL.Image. The use of `load_img` requires PIL."
        )
    if isinstance(path, io.BytesIO):
        img = pil_image.open(path)
    elif isinstance(path, (pathlib.Path, bytes, str)):
        if isinstance(path, pathlib.Path):
            path = str(path.resolve())
        with open(path, "rb") as f:
            img = pil_image.open(io.BytesIO(f.read()))
    else:
        raise TypeError(
            f"path should be path-like or io.BytesIO, not {type(path)}"
        )

    if color_mode == "grayscale":
        # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
        # convert it to an 8-bit grayscale image.
        if img.mode not in ("L", "I;16", "I"):
            img = img.convert("L")
    elif color_mode == "rgba":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    elif color_mode == "rgb":
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    "Invalid interpolation method {} specified. Supported "
                    "methods are {}".format(
                        interpolation,
                        ", ".join(PIL_INTERPOLATION_METHODS.keys()),
                    )
                )
            resample = PIL_INTERPOLATION_METHODS[interpolation]

            if keep_aspect_ratio:
                width, height = img.size
                target_width, target_height = width_height_tuple

                crop_height = (width * target_height) // target_width
                crop_width = (height * target_width) // target_height

                # Set back to input height / width
                # if crop_height / crop_width is not smaller.
                crop_height = min(height, crop_height)
                crop_width = min(width, crop_width)

                crop_box_hstart = (height - crop_height) // 2
                crop_box_wstart = (width - crop_width) // 2
                crop_box_wend = crop_box_wstart + crop_width
                crop_box_hend = crop_box_hstart + crop_height
                crop_box = [
                    crop_box_wstart,
                    crop_box_hstart,
                    crop_box_wend,
                    crop_box_hend,
                ]
                img = img.resize(width_height_tuple, resample, box=crop_box)
            else:
                img = img.resize(width_height_tuple, resample)
    return img


@keras_export("keras.preprocessing.image.smart_resize")
def smart_resize(
    x,
    size,
    interpolation="bilinear",
    data_format="channels_last",
    backend_module=None,
):
    """Resize images to a target size without aspect ratio distortion.

    Image datasets typically yield images that have each a different
    size. However, these images need to be batched before they can be
    processed by Keras layers. To be batched, images need to share the same
    height and width.

    You could simply do, in TF (or JAX equivalent):

    ```python
    size = (200, 200)
    ds = ds.map(lambda img: resize(img, size))
    ```

    However, if you do this, you distort the aspect ratio of your images, since
    in general they do not all have the same aspect ratio as `size`. This is
    fine in many cases, but not always (e.g. for image generation models
    this can be a problem).

    Note that passing the argument `preserve_aspect_ratio=True` to `resize`
    will preserve the aspect ratio, but at the cost of no longer respecting the
    provided target size.

    This calls for:

    ```python
    size = (200, 200)
    ds = ds.map(lambda img: smart_resize(img, size))
    ```

    Your output images will actually be `(200, 200)`, and will not be distorted.
    Instead, the parts of the image that do not fit within the target size
    get cropped out.

    The resizing process is:

    1. Take the largest centered crop of the image that has the same aspect
    ratio as the target size. For instance, if `size=(200, 200)` and the input
    image has size `(340, 500)`, we take a crop of `(340, 340)` centered along
    the width.
    2. Resize the cropped image to the target size. In the example above,
    we resize the `(340, 340)` crop to `(200, 200)`.

    Args:
        x: Input image or batch of images (as a tensor or NumPy array).
            Must be in format `(height, width, channels)`
            or `(batch_size, height, width, channels)`.
        size: Tuple of `(height, width)` integer. Target size.
        interpolation: String, interpolation to use for resizing.
            Defaults to `'bilinear'`.
            Supports `bilinear`, `nearest`, `bicubic`,
            `lanczos3`, `lanczos5`.
        data_format: `"channels_last"` or `"channels_first"`.
        backend_module: Backend module to use (if different from the default
            backend).

    Returns:
        Array with shape `(size[0], size[1], channels)`.
        If the input image was a NumPy array, the output is a NumPy array,
        and if it was a backend-native tensor,
        the output is a backend-native tensor.
    """
    backend_module = backend_module or backend
    if len(size) != 2:
        raise ValueError(
            f"Expected `size` to be a tuple of 2 integers, but got: {size}."
        )
    img = backend_module.convert_to_tensor(x)
    if len(img.shape) is not None:
        if len(img.shape) < 3 or len(img.shape) > 4:
            raise ValueError(
                "Expected an image array with shape `(height, width, "
                "channels)`, or `(batch_size, height, width, channels)`, but "
                f"got input with incorrect rank, of shape {img.shape}."
            )
    shape = backend_module.shape(img)
    if data_format == "channels_last":
        height, width = shape[-3], shape[-2]
    else:
        height, width = shape[-2], shape[-1]
    target_height, target_width = size

    # Set back to input height / width if crop_height / crop_width is not
    # smaller.
    if isinstance(height, int) and isinstance(width, int):
        # For JAX, we need to keep the slice indices as static integers
        crop_height = int(float(width * target_height) / target_width)
        crop_height = min(height, crop_height)
        crop_width = int(float(height * target_width) / target_height)
        crop_width = min(width, crop_width)
        crop_box_hstart = int(float(height - crop_height) / 2)
        crop_box_wstart = int(float(width - crop_width) / 2)
    else:
        crop_height = backend_module.cast(
            backend_module.cast(width * target_height, "float32")
            / target_width,
            "int32",
        )
        crop_height = backend_module.numpy.minimum(height, crop_height)
        crop_height = backend_module.cast(crop_height, "int32")
        crop_width = backend_module.cast(
            backend_module.cast(height * target_width, "float32")
            / target_height,
            "int32",
        )
        crop_width = backend_module.numpy.minimum(width, crop_width)
        crop_width = backend_module.cast(crop_width, "int32")

        crop_box_hstart = backend_module.cast(
            backend_module.cast(height - crop_height, "float32") / 2, "int32"
        )
        crop_box_wstart = backend_module.cast(
            backend_module.cast(width - crop_width, "float32") / 2, "int32"
        )

    if data_format == "channels_last":
        if len(img.shape) == 4:
            img = img[
                :,
                crop_box_hstart : crop_box_hstart + crop_height,
                crop_box_wstart : crop_box_wstart + crop_width,
                :,
            ]
        else:
            img = img[
                crop_box_hstart : crop_box_hstart + crop_height,
                crop_box_wstart : crop_box_wstart + crop_width,
                :,
            ]
    else:
        if len(img.shape) == 4:
            img = img[
                :,
                :,
                crop_box_hstart : crop_box_hstart + crop_height,
                crop_box_wstart : crop_box_wstart + crop_width,
            ]
        else:
            img = img[
                :,
                crop_box_hstart : crop_box_hstart + crop_height,
                crop_box_wstart : crop_box_wstart + crop_width,
            ]

    img = backend_module.image.resize(
        img, size=size, interpolation=interpolation, data_format=data_format
    )

    if isinstance(x, np.ndarray):
        return np.array(img)
    return img
