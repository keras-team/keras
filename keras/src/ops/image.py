from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import compute_conv_output_shape


class RGBToGrayscale(Operation):
    def __init__(self, data_format=None):
        super().__init__()
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return backend.image.rgb_to_grayscale(
            images, data_format=self.data_format
        )

    def compute_output_spec(self, images):
        images_shape = list(images.shape)
        if len(images_shape) not in (3, 4):
            raise ValueError(
                "Invalid images rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). "
                f"Received: images.shape={images_shape}"
            )
        if self.data_format == "channels_last":
            images_shape[-1] = 1
        else:
            images_shape[-3] = 1
        return KerasTensor(shape=images_shape, dtype=images.dtype)


@keras_export("keras.ops.image.rgb_to_grayscale")
def rgb_to_grayscale(images, data_format=None):
    """Convert RGB images to grayscale.

    This function converts RGB images to grayscale images. It supports both
    3D and 4D tensors.

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Grayscale image or batch of grayscale images.

    Examples:

    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.random.random((2, 4, 4, 3))
    >>> y = ops.image.rgb_to_grayscale(x)
    >>> y.shape
    (2, 4, 4, 1)

    >>> x = np.random.random((4, 4, 3)) # Single RGB image
    >>> y = ops.image.rgb_to_grayscale(x)
    >>> y.shape
    (4, 4, 1)

    >>> x = np.random.random((2, 3, 4, 4))
    >>> y = ops.image.rgb_to_grayscale(x, data_format="channels_first")
    >>> y.shape
    (2, 1, 4, 4)
    """
    if any_symbolic_tensors((images,)):
        return RGBToGrayscale(data_format=data_format).symbolic_call(images)
    return backend.image.rgb_to_grayscale(images, data_format=data_format)


class RGBToHSV(Operation):
    def __init__(self, data_format=None):
        super().__init__()
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return backend.image.rgb_to_hsv(images, data_format=self.data_format)

    def compute_output_spec(self, images):
        images_shape = list(images.shape)
        dtype = images.dtype
        if len(images_shape) not in (3, 4):
            raise ValueError(
                "Invalid images rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). "
                f"Received: images.shape={images_shape}"
            )
        if not backend.is_float_dtype(dtype):
            raise ValueError(
                "Invalid images dtype: expected float dtype. "
                f"Received: images.dtype={dtype}"
            )
        return KerasTensor(shape=images_shape, dtype=images.dtype)


@keras_export("keras.ops.image.rgb_to_hsv")
def rgb_to_hsv(images, data_format=None):
    """Convert RGB images to HSV.

    `images` must be of float dtype, and the output is only well defined if the
    values in `images` are in `[0, 1]`.

    All HSV values are in `[0, 1]`. A hue of `0` corresponds to pure red, `1/3`
    is pure green, and `2/3` is pure blue.

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        HSV image or batch of HSV images.

    Examples:

    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.random.random((2, 4, 4, 3))
    >>> y = ops.image.rgb_to_hsv(x)
    >>> y.shape
    (2, 4, 4, 3)

    >>> x = np.random.random((4, 4, 3)) # Single RGB image
    >>> y = ops.image.rgb_to_hsv(x)
    >>> y.shape
    (4, 4, 3)

    >>> x = np.random.random((2, 3, 4, 4))
    >>> y = ops.image.rgb_to_hsv(x, data_format="channels_first")
    >>> y.shape
    (2, 3, 4, 4)
    """
    if any_symbolic_tensors((images,)):
        return RGBToHSV(data_format=data_format).symbolic_call(images)
    return backend.image.rgb_to_hsv(images, data_format=data_format)


class HSVToRGB(Operation):
    def __init__(self, data_format=None):
        super().__init__()
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return backend.image.hsv_to_rgb(images, data_format=self.data_format)

    def compute_output_spec(self, images):
        images_shape = list(images.shape)
        dtype = images.dtype
        if len(images_shape) not in (3, 4):
            raise ValueError(
                "Invalid images rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). "
                f"Received: images.shape={images_shape}"
            )
        if not backend.is_float_dtype(dtype):
            raise ValueError(
                "Invalid images dtype: expected float dtype. "
                f"Received: images.dtype={dtype}"
            )
        return KerasTensor(shape=images_shape, dtype=images.dtype)


@keras_export("keras.ops.image.hsv_to_rgb")
def hsv_to_rgb(images, data_format=None):
    """Convert HSV images to RGB.

    `images` must be of float dtype, and the output is only well defined if the
    values in `images` are in `[0, 1]`.

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        RGB image or batch of RGB images.

    Examples:

    >>> import numpy as np
    >>> from keras import ops
    >>> x = np.random.random((2, 4, 4, 3))
    >>> y = ops.image.hsv_to_rgb(x)
    >>> y.shape
    (2, 4, 4, 3)

    >>> x = np.random.random((4, 4, 3)) # Single HSV image
    >>> y = ops.image.hsv_to_rgb(x)
    >>> y.shape
    (4, 4, 3)

    >>> x = np.random.random((2, 3, 4, 4))
    >>> y = ops.image.hsv_to_rgb(x, data_format="channels_first")
    >>> y.shape
    (2, 3, 4, 4)
    """
    if any_symbolic_tensors((images,)):
        return HSVToRGB(data_format=data_format).symbolic_call(images)
    return backend.image.hsv_to_rgb(images, data_format=data_format)


class Resize(Operation):
    def __init__(
        self,
        size,
        interpolation="bilinear",
        antialias=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        fill_mode="constant",
        fill_value=0.0,
        data_format=None,
    ):
        super().__init__()
        self.size = tuple(size)
        self.interpolation = interpolation
        self.antialias = antialias
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.pad_to_aspect_ratio = pad_to_aspect_ratio
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return _resize(
            images,
            self.size,
            interpolation=self.interpolation,
            antialias=self.antialias,
            data_format=self.data_format,
            crop_to_aspect_ratio=self.crop_to_aspect_ratio,
            pad_to_aspect_ratio=self.pad_to_aspect_ratio,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )

    def compute_output_spec(self, images):
        images_shape = list(images.shape)
        if len(images_shape) not in (3, 4):
            raise ValueError(
                "Invalid images rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"images.shape={images.shape}"
            )
        if self.data_format == "channels_last":
            height_axis, width_axis = -3, -2
        else:
            height_axis, width_axis = -2, -1
        images_shape[height_axis] = self.size[0]
        images_shape[width_axis] = self.size[1]
        return KerasTensor(shape=images_shape, dtype=images.dtype)


@keras_export("keras.ops.image.resize")
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
    """Resize images to size using the specified interpolation method.

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        size: Size of output image in `(height, width)` format.
        interpolation: Interpolation method. Available methods are `"nearest"`,
            `"bilinear"`, and `"bicubic"`. Defaults to `"bilinear"`.
        antialias: Whether to use an antialiasing filter when downsampling an
            image. Defaults to `False`.
        crop_to_aspect_ratio: If `True`, resize the images without aspect
            ratio distortion. When the original aspect ratio differs
            from the target aspect ratio, the output image will be
            cropped so as to return the
            largest possible window in the image (of size `(height, width)`)
            that matches the target aspect ratio. By default
            (`crop_to_aspect_ratio=False`), aspect ratio may not be preserved.
        pad_to_aspect_ratio: If `True`, pad the images without aspect
            ratio distortion. When the original aspect ratio differs
            from the target aspect ratio, the output image will be
            evenly padded on the short side.
        fill_mode: When using `pad_to_aspect_ratio=True`, padded areas
            are filled according to the given mode. Only `"constant"` is
            supported at this time
            (fill with constant value, equal to `fill_value`).
        fill_value: Float. Padding value to use when `pad_to_aspect_ratio=True`.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Resized image or batch of images.

    Examples:

    >>> x = np.random.random((2, 4, 4, 3)) # batch of 2 RGB images
    >>> y = keras.ops.image.resize(x, (2, 2))
    >>> y.shape
    (2, 2, 2, 3)

    >>> x = np.random.random((4, 4, 3)) # single RGB image
    >>> y = keras.ops.image.resize(x, (2, 2))
    >>> y.shape
    (2, 2, 3)

    >>> x = np.random.random((2, 3, 4, 4)) # batch of 2 RGB images
    >>> y = keras.ops.image.resize(x, (2, 2),
    ...     data_format="channels_first")
    >>> y.shape
    (2, 3, 2, 2)
    """
    if len(size) != 2:
        raise ValueError(
            "Expected `size` to be a tuple of 2 integers. "
            f"Received: size={size}"
        )
    if len(images.shape) < 3 or len(images.shape) > 4:
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if pad_to_aspect_ratio and crop_to_aspect_ratio:
        raise ValueError(
            "Only one of `pad_to_aspect_ratio` & `crop_to_aspect_ratio` "
            "can be `True`."
        )
    if any_symbolic_tensors((images,)):
        return Resize(
            size,
            interpolation=interpolation,
            antialias=antialias,
            data_format=data_format,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            fill_mode=fill_mode,
            fill_value=fill_value,
        ).symbolic_call(images)
    return _resize(
        images,
        size,
        interpolation=interpolation,
        antialias=antialias,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
        data_format=data_format,
        pad_to_aspect_ratio=pad_to_aspect_ratio,
        fill_mode=fill_mode,
        fill_value=fill_value,
    )


def _resize(
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
    resized = backend.image.resize(
        images,
        size,
        interpolation=interpolation,
        antialias=antialias,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
        data_format=data_format,
        pad_to_aspect_ratio=pad_to_aspect_ratio,
        fill_mode=fill_mode,
        fill_value=fill_value,
    )
    if resized.dtype == images.dtype:
        # Only `torch` backend will cast result to original dtype with
        # correct rounding and without dtype overflow
        return resized
    if backend.is_int_dtype(images.dtype):
        resized = ops.round(resized)
    return ops.saturate_cast(resized, images.dtype)


class AffineTransform(Operation):
    def __init__(
        self,
        interpolation="bilinear",
        fill_mode="constant",
        fill_value=0,
        data_format=None,
    ):
        super().__init__()
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images, transform):
        return backend.image.affine_transform(
            images,
            transform,
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            data_format=self.data_format,
        )

    def compute_output_spec(self, images, transform):
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
        return KerasTensor(images.shape, dtype=images.dtype)


@keras_export("keras.ops.image.affine_transform")
def affine_transform(
    images,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format=None,
):
    """Applies the given transform(s) to the image(s).

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        transform: Projective transform matrix/matrices. A vector of length 8 or
            tensor of size N x 8. If one row of transform is
            `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps the output point
            `(x, y)` to a transformed input point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
            where `k = c0 x + c1 y + 1`. The transform is inverted compared to
            the transform mapping input points to output points. Note that
            gradients are not backpropagated into transformation parameters.
            Note that `c0` and `c1` are only effective when using TensorFlow
            backend and will be considered as `0` when using other backends.
        interpolation: Interpolation method. Available methods are `"nearest"`,
            and `"bilinear"`. Defaults to `"bilinear"`.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode. Available methods are `"constant"`,
            `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.
            - `"reflect"`: `(d c b a | a b c d | d c b a)`
                The input is extended by reflecting about the edge of the last
                pixel.
            - `"constant"`: `(k k k k | a b c d | k k k k)`
                The input is extended by filling all values beyond
                the edge with the same constant value k specified by
                `fill_value`.
            - `"wrap"`: `(a b c d | a b c d | a b c d)`
                The input is extended by wrapping around to the opposite edge.
            - `"nearest"`: `(a a a a | a b c d | d d d d)`
                The input is extended by the nearest pixel.
        fill_value: Value used for points outside the boundaries of the input if
            `fill_mode="constant"`. Defaults to `0`.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Applied affine transform image or batch of images.

    Examples:

    >>> x = np.random.random((2, 64, 80, 3)) # batch of 2 RGB images
    >>> transform = np.array(
    ...     [
    ...         [1.5, 0, -20, 0, 1.5, -16, 0, 0],  # zoom
    ...         [1, 0, -20, 0, 1, -16, 0, 0],  # translation
    ...     ]
    ... )
    >>> y = keras.ops.image.affine_transform(x, transform)
    >>> y.shape
    (2, 64, 80, 3)

    >>> x = np.random.random((64, 80, 3)) # single RGB image
    >>> transform = np.array([1.0, 0.5, -20, 0.5, 1.0, -16, 0, 0])  # shear
    >>> y = keras.ops.image.affine_transform(x, transform)
    >>> y.shape
    (64, 80, 3)

    >>> x = np.random.random((2, 3, 64, 80)) # batch of 2 RGB images
    >>> transform = np.array(
    ...     [
    ...         [1.5, 0, -20, 0, 1.5, -16, 0, 0],  # zoom
    ...         [1, 0, -20, 0, 1, -16, 0, 0],  # translation
    ...     ]
    ... )
    >>> y = keras.ops.image.affine_transform(x, transform,
    ...     data_format="channels_first")
    >>> y.shape
    (2, 3, 64, 80)
    """
    if any_symbolic_tensors((images, transform)):
        return AffineTransform(
            interpolation=interpolation,
            fill_mode=fill_mode,
            fill_value=fill_value,
            data_format=data_format,
        ).symbolic_call(images, transform)
    return backend.image.affine_transform(
        images,
        transform,
        interpolation=interpolation,
        fill_mode=fill_mode,
        fill_value=fill_value,
        data_format=data_format,
    )


class ExtractPatches(Operation):
    def __init__(
        self,
        size,
        strides=None,
        dilation_rate=1,
        padding="valid",
        data_format=None,
    ):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return _extract_patches(
            images=images,
            size=self.size,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            padding=self.padding,
            data_format=self.data_format,
        )

    def compute_output_spec(self, images):
        images_shape = list(images.shape)
        original_ndim = len(images_shape)
        if not self.strides:
            strides = (self.size[0], self.size[1])
        if self.data_format == "channels_last":
            channels_in = images_shape[-1]
        else:
            channels_in = images_shape[-3]
        if original_ndim == 3:
            images_shape = [1] + images_shape
        filters = self.size[0] * self.size[1] * channels_in
        kernel_size = (self.size[0], self.size[1])
        out_shape = compute_conv_output_shape(
            images_shape,
            filters,
            kernel_size,
            strides=strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if original_ndim == 3:
            out_shape = out_shape[1:]
        return KerasTensor(shape=out_shape, dtype=images.dtype)


@keras_export("keras.ops.image.extract_patches")
def extract_patches(
    images,
    size,
    strides=None,
    dilation_rate=1,
    padding="valid",
    data_format=None,
):
    """Extracts patches from the image(s).

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        size: Patch size int or tuple (patch_height, patch_width)
        strides: strides along height and width. If not specified, or
            if `None`, it defaults to the same value as `size`.
        dilation_rate: This is the input stride, specifying how far two
            consecutive patch samples are in the input. For value other than 1,
            strides must be 1. NOTE: `strides > 1` is not supported in
            conjunction with `dilation_rate > 1`
        padding: The type of padding algorithm to use: `"same"` or `"valid"`.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Extracted patches 3D (if not batched) or 4D (if batched)

    Examples:

    >>> image = np.random.random(
    ...     (2, 20, 20, 3)
    ... ).astype("float32") # batch of 2 RGB images
    >>> patches = keras.ops.image.extract_patches(image, (5, 5))
    >>> patches.shape
    (2, 4, 4, 75)
    >>> image = np.random.random((20, 20, 3)).astype("float32") # 1 RGB image
    >>> patches = keras.ops.image.extract_patches(image, (3, 3), (1, 1))
    >>> patches.shape
    (18, 18, 27)
    """
    if any_symbolic_tensors((images,)):
        return ExtractPatches(
            size=size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            data_format=data_format,
        ).symbolic_call(images)

    return _extract_patches(
        images, size, strides, dilation_rate, padding, data_format=data_format
    )


def _extract_patches(
    images,
    size,
    strides=None,
    dilation_rate=1,
    padding="valid",
    data_format=None,
):
    if isinstance(size, int):
        patch_h = patch_w = size
    elif len(size) == 2:
        patch_h, patch_w = size[0], size[1]
    else:
        raise TypeError(
            "Invalid `size` argument. Expected an "
            f"int or a tuple of length 2. Received: size={size}"
        )
    data_format = backend.standardize_data_format(data_format)
    if data_format == "channels_last":
        channels_in = images.shape[-1]
    elif data_format == "channels_first":
        channels_in = images.shape[-3]
    if not strides:
        strides = size
    out_dim = patch_h * patch_w * channels_in
    kernel = backend.numpy.eye(out_dim, dtype=images.dtype)
    kernel = backend.numpy.reshape(
        kernel, (patch_h, patch_w, channels_in, out_dim)
    )
    _unbatched = False
    if len(images.shape) == 3:
        _unbatched = True
        images = backend.numpy.expand_dims(images, axis=0)
    patches = backend.nn.conv(
        inputs=images,
        kernel=kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
    )
    if _unbatched:
        patches = backend.numpy.squeeze(patches, axis=0)
    return patches


class MapCoordinates(Operation):
    def __init__(self, order, fill_mode="constant", fill_value=0):
        super().__init__()
        self.order = order
        self.fill_mode = fill_mode
        self.fill_value = fill_value

    def call(self, inputs, coordinates):
        return backend.image.map_coordinates(
            inputs,
            coordinates,
            order=self.order,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )

    def compute_output_spec(self, inputs, coordinates):
        if coordinates.shape[0] != len(inputs.shape):
            raise ValueError(
                "First dim of `coordinates` must be the same as the rank of "
                "`inputs`. "
                f"Received inputs with shape: {inputs.shape} and coordinate "
                f"leading dim of {coordinates.shape[0]}"
            )
        if len(coordinates.shape) < 2:
            raise ValueError(
                "Invalid coordinates rank: expected at least rank 2."
                f" Received input with shape: {coordinates.shape}"
            )
        return KerasTensor(coordinates.shape[1:], dtype=inputs.dtype)


@keras_export("keras.ops.image.map_coordinates")
def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0
):
    """Map the input array to new coordinates by interpolation.

    Note that interpolation near boundaries differs from the scipy function,
    because we fixed an outstanding bug
    [scipy/issues/2640](https://github.com/scipy/scipy/issues/2640).

    Args:
        inputs: The input array.
        coordinates: The coordinates at which inputs is evaluated.
        order: The order of the spline interpolation. The order must be `0` or
            `1`. `0` indicates the nearest neighbor and `1` indicates the linear
            interpolation.
        fill_mode: Points outside the boundaries of the inputs are filled
            according to the given mode. Available methods are `"constant"`,
            `"nearest"`, `"wrap"` and `"mirror"` and `"reflect"`. Defaults to
            `"constant"`.
            - `"constant"`: `(k k k k | a b c d | k k k k)`
                The inputs is extended by filling all values beyond
                the edge with the same constant value k specified by
                `fill_value`.
            - `"nearest"`: `(a a a a | a b c d | d d d d)`
                The inputs is extended by the nearest pixel.
            - `"wrap"`: `(a b c d | a b c d | a b c d)`
                The inputs is extended by wrapping around to the opposite edge.
            - `"mirror"`: `(c d c b | a b c d | c b a b)`
                The inputs is extended by mirroring about the edge.
            - `"reflect"`: `(d c b a | a b c d | d c b a)`
                The inputs is extended by reflecting about the edge of the last
                pixel.
        fill_value: Value used for points outside the boundaries of the inputs
            if `fill_mode="constant"`. Defaults to `0`.

    Returns:
        Output input or batch of inputs.

    """
    if any_symbolic_tensors((inputs, coordinates)):
        return MapCoordinates(
            order,
            fill_mode,
            fill_value,
        ).symbolic_call(inputs, coordinates)
    return backend.image.map_coordinates(
        inputs,
        coordinates,
        order,
        fill_mode,
        fill_value,
    )


class PadImages(Operation):
    def __init__(
        self,
        top_padding=None,
        left_padding=None,
        bottom_padding=None,
        right_padding=None,
        target_height=None,
        target_width=None,
        data_format=None,
    ):
        super().__init__()
        self.top_padding = top_padding
        self.left_padding = left_padding
        self.bottom_padding = bottom_padding
        self.right_padding = right_padding
        self.target_height = target_height
        self.target_width = target_width
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return _pad_images(
            images,
            self.top_padding,
            self.left_padding,
            self.bottom_padding,
            self.right_padding,
            self.target_height,
            self.target_width,
            self.data_format,
        )

    def compute_output_spec(self, images):
        images_shape = list(images.shape)

        if self.data_format == "channels_last":
            height_axis, width_axis = -3, -2
            height, width = images_shape[height_axis], images_shape[width_axis]
        else:
            height_axis, width_axis = -2, -1
            height, width = images_shape[height_axis], images_shape[width_axis]

        target_height = self.target_height
        if target_height is None and height is not None:
            target_height = self.top_padding + height + self.bottom_padding
        target_width = self.target_width
        if target_width is None and width is not None:
            target_width = self.left_padding + width + self.right_padding

        images_shape[height_axis] = target_height
        images_shape[width_axis] = target_width
        return KerasTensor(shape=images_shape, dtype=images.dtype)


@keras_export("keras.ops.image.pad_images")
def pad_images(
    images,
    top_padding=None,
    left_padding=None,
    bottom_padding=None,
    right_padding=None,
    target_height=None,
    target_width=None,
    data_format=None,
):
    """Pad `images` with zeros to the specified `height` and `width`.

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        top_padding: Number of rows of zeros to add on top.
        left_padding: Number of columns of zeros to add on the left.
        bottom_padding: Number of rows of zeros to add at the bottom.
        right_padding: Number of columns of zeros to add on the right.
        target_height: Height of output images.
        target_width: Width of output images.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Padded image or batch of images.

    Example:

    >>> images = np.random.random((15, 25, 3))
    >>> padded_images = keras.ops.image.pad_images(
    ...     images, 2, 3, target_height=20, target_width=30
    ... )
    >>> padded_images.shape
    (20, 30, 3)

    >>> batch_images = np.random.random((2, 15, 25, 3))
    >>> padded_batch = keras.ops.image.pad_images(
    ...     batch_images, 2, 3, target_height=20, target_width=30
    ... )
    >>> padded_batch.shape
    (2, 20, 30, 3)"""

    if any_symbolic_tensors((images,)):
        return PadImages(
            top_padding,
            left_padding,
            bottom_padding,
            right_padding,
            target_height,
            target_width,
            data_format,
        ).symbolic_call(images)

    return _pad_images(
        images,
        top_padding,
        left_padding,
        bottom_padding,
        right_padding,
        target_height,
        target_width,
        data_format,
    )


def _pad_images(
    images,
    top_padding,
    left_padding,
    bottom_padding,
    right_padding,
    target_height,
    target_width,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    images = backend.convert_to_tensor(images)
    images_shape = ops.shape(images)

    # Check
    if len(images_shape) not in (3, 4):
        raise ValueError(
            f"Invalid shape for argument `images`: "
            "it must have rank 3 or 4. "
            f"Received: images.shape={images_shape}"
        )
    if [top_padding, bottom_padding, target_height].count(None) != 1:
        raise ValueError(
            "Must specify exactly two of "
            "top_padding, bottom_padding, target_height. "
            f"Received: top_padding={top_padding}, "
            f"bottom_padding={bottom_padding}, "
            f"target_height={target_height}"
        )
    if [left_padding, right_padding, target_width].count(None) != 1:
        raise ValueError(
            "Must specify exactly two of "
            "left_padding, right_padding, target_width. "
            f"Received: left_padding={left_padding}, "
            f"right_padding={right_padding}, "
            f"target_width={target_width}"
        )

    is_batch = False if len(images_shape) == 3 else True
    if data_format == "channels_last":
        height, width = images_shape[-3], images_shape[-2]
    else:
        height, width = images_shape[-2], images_shape[-1]

    # Infer padding
    if top_padding is None:
        top_padding = target_height - bottom_padding - height
    if bottom_padding is None:
        bottom_padding = target_height - top_padding - height
    if left_padding is None:
        left_padding = target_width - right_padding - width
    if right_padding is None:
        right_padding = target_width - left_padding - width

    if top_padding < 0:
        raise ValueError(
            f"top_padding must be >= 0. Received: top_padding={top_padding}"
        )
    if left_padding < 0:
        raise ValueError(
            "left_padding must be >= 0. "
            f"Received: left_padding={left_padding}"
        )
    if right_padding < 0:
        raise ValueError(
            "right_padding must be >= 0. "
            f"Received: right_padding={right_padding}"
        )
    if bottom_padding < 0:
        raise ValueError(
            "bottom_padding must be >= 0. "
            f"Received: bottom_padding={bottom_padding}"
        )

    # Compute pad_width
    pad_width = [[top_padding, bottom_padding], [left_padding, right_padding]]
    if data_format == "channels_last":
        pad_width = pad_width + [[0, 0]]
    else:
        pad_width = [[0, 0]] + pad_width
    if is_batch:
        pad_width = [[0, 0]] + pad_width

    padded_images = backend.numpy.pad(images, pad_width)
    return padded_images


class CropImages(Operation):
    def __init__(
        self,
        top_cropping,
        left_cropping,
        bottom_cropping,
        right_cropping,
        target_height,
        target_width,
        data_format=None,
    ):
        super().__init__()
        self.top_cropping = top_cropping
        self.bottom_cropping = bottom_cropping
        self.left_cropping = left_cropping
        self.right_cropping = right_cropping
        self.target_height = target_height
        self.target_width = target_width
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return _crop_images(
            images,
            self.top_cropping,
            self.left_cropping,
            self.bottom_cropping,
            self.right_cropping,
            self.target_height,
            self.target_width,
            self.data_format,
        )

    def compute_output_spec(self, images):
        images_shape = list(images.shape)

        if self.data_format == "channels_last":
            height_axis, width_axis = -3, -2
        else:
            height_axis, width_axis = -2, -1
        height, width = images_shape[height_axis], images_shape[width_axis]

        if height is None and self.target_height is None:
            raise ValueError(
                "When the height of the images is unknown, `target_height` "
                "must be specified."
                f"Received images.shape={images_shape} and "
                f"target_height={self.target_height}"
            )
        if width is None and self.target_width is None:
            raise ValueError(
                "When the width of the images is unknown, `target_width` "
                "must be specified."
                f"Received images.shape={images_shape} and "
                f"target_width={self.target_width}"
            )

        target_height = self.target_height
        if target_height is None:
            target_height = height - self.top_cropping - self.bottom_cropping
        target_width = self.target_width
        if target_width is None:
            target_width = width - self.left_cropping - self.right_cropping

        images_shape[height_axis] = target_height
        images_shape[width_axis] = target_width
        return KerasTensor(shape=images_shape, dtype=images.dtype)


@keras_export("keras.ops.image.crop_images")
def crop_images(
    images,
    top_cropping=None,
    left_cropping=None,
    bottom_cropping=None,
    right_cropping=None,
    target_height=None,
    target_width=None,
    data_format=None,
):
    """Crop `images` to a specified `height` and `width`.

    Args:
        images: Input image or batch of images. Must be 3D or 4D.
        top_cropping: Number of columns to crop from the top.
        left_cropping: Number of columns to crop from the left.
        bottom_cropping: Number of columns to crop from the bottom.
        right_cropping: Number of columns to crop from the right.
        target_height: Height of the output images.
        target_width: Width of the output images.
        data_format: A string specifying the data format of the input tensor.
            It can be either `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)`, while `"channels_first"`
            corresponds to inputs with shape `(batch, channels, height, width)`.
            If not specified, the value will default to
            `keras.config.image_data_format`.

    Returns:
        Cropped image or batch of images.

    Example:

    >>> images = np.reshape(np.arange(1, 28, dtype="float32"), [3, 3, 3])
    >>> images[:,:,0] # print the first channel of the images
    array([[ 1.,  4.,  7.],
           [10., 13., 16.],
           [19., 22., 25.]], dtype=float32)
    >>> cropped_images = keras.image.crop_images(images, 0, 0, 2, 2)
    >>> cropped_images[:,:,0] # print the first channel of the cropped images
    array([[ 1.,  4.],
           [10., 13.]], dtype=float32)"""

    if any_symbolic_tensors((images,)):
        return CropImages(
            top_cropping,
            left_cropping,
            bottom_cropping,
            right_cropping,
            target_height,
            target_width,
            data_format,
        ).symbolic_call(images)

    return _crop_images(
        images,
        top_cropping,
        left_cropping,
        bottom_cropping,
        right_cropping,
        target_height,
        target_width,
        data_format,
    )


def _crop_images(
    images,
    top_cropping,
    left_cropping,
    bottom_cropping,
    right_cropping,
    target_height,
    target_width,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    images = backend.convert_to_tensor(images)
    images_shape = ops.shape(images)

    # Check
    if len(images_shape) not in (3, 4):
        raise ValueError(
            f"Invalid shape for argument `images`: "
            "it must have rank 3 or 4. "
            f"Received: images.shape={images_shape}"
        )
    if [top_cropping, bottom_cropping, target_height].count(None) != 1:
        raise ValueError(
            "Must specify exactly two of "
            "top_cropping, bottom_cropping, target_height. "
            f"Received: top_cropping={top_cropping}, "
            f"bottom_cropping={bottom_cropping}, "
            f"target_height={target_height}"
        )
    if [left_cropping, right_cropping, target_width].count(None) != 1:
        raise ValueError(
            "Must specify exactly two of "
            "left_cropping, right_cropping, target_width. "
            f"Received: left_cropping={left_cropping}, "
            f"right_cropping={right_cropping}, "
            f"target_width={target_width}"
        )

    is_batch = False if len(images_shape) == 3 else True
    if data_format == "channels_last":
        height, width = images_shape[-3], images_shape[-2]
        channels = images_shape[-1]
    else:
        height, width = images_shape[-2], images_shape[-1]
        channels = images_shape[-3]

    # Infer padding
    if top_cropping is None:
        top_cropping = height - target_height - bottom_cropping
    if target_height is None:
        target_height = height - bottom_cropping - top_cropping
    if left_cropping is None:
        left_cropping = width - target_width - right_cropping
    if target_width is None:
        target_width = width - right_cropping - left_cropping

    if top_cropping < 0:
        raise ValueError(
            "top_cropping must be >= 0. "
            f"Received: top_cropping={top_cropping}"
        )
    if target_height < 0:
        raise ValueError(
            "target_height must be >= 0. "
            f"Received: target_height={target_height}"
        )
    if left_cropping < 0:
        raise ValueError(
            "left_cropping must be >= 0. "
            f"Received: left_cropping={left_cropping}"
        )
    if target_width < 0:
        raise ValueError(
            "target_width must be >= 0. "
            f"Received: target_width={target_width}"
        )

    # Compute start_indices and shape
    start_indices = [top_cropping, left_cropping]
    shape = [target_height, target_width]
    if data_format == "channels_last":
        start_indices = start_indices + [0]
        shape = shape + [channels]
    else:
        start_indices = [0] + start_indices
        shape = [channels] + shape
    if is_batch:
        batch_size = images_shape[0]
        start_indices = [0] + start_indices
        shape = [batch_size] + shape

    cropped_images = ops.slice(images, start_indices, shape)
    return cropped_images
