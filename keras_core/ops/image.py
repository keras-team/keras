from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.ops.operation import Operation


class Resize(Operation):
    def __init__(
        self,
        size,
        interpolation="bilinear",
        antialias=False,
        data_format="channels_last",
    ):
        super().__init__()
        self.size = tuple(size)
        self.interpolation = interpolation
        self.antialias = antialias
        self.data_format = data_format

    def call(self, image):
        return backend.image.resize(
            image,
            self.size,
            interpolation=self.interpolation,
            antialias=self.antialias,
            data_format=self.data_format,
        )

    def compute_output_spec(self, image):
        if len(image.shape) == 3:
            return KerasTensor(
                self.size + (image.shape[-1],), dtype=image.dtype
            )
        elif len(image.shape) == 4:
            if self.data_format == "channels_last":
                return KerasTensor(
                    (image.shape[0],) + self.size + (image.shape[-1],),
                    dtype=image.dtype,
                )
            else:
                return KerasTensor(
                    (image.shape[0], image.shape[1]) + self.size,
                    dtype=image.dtype,
                )
        raise ValueError(
            "Invalid input rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"image.shape={image.shape}"
        )


@keras_core_export("keras_core.ops.image.resize")
def resize(
    image,
    size,
    interpolation="bilinear",
    antialias=False,
    data_format="channels_last",
):
    """Resize images to size using the specified interpolation method.

    Args:
        image: Input image or batch of images. Must be 3D or 4D.
        size: Size of output image in `(height, width)` format.
        interpolation: Interpolation method. Available methods are `"nearest"`,
            `"bilinear"`, and `"bicubic"`. Defaults to `"bilinear"`.
        antialias: Whether to use an antialiasing filter when downsampling an
            image. Defaults to `False`.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, weight)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.

    Returns:
        Resized image or batch of images.

    Examples:

    >>> x = np.random.random((2, 4, 4, 3)) # batch of 2 RGB images
    >>> y = keras_core.ops.image.resize(x, (2, 2))
    >>> y.shape
    (2, 2, 2, 3)

    >>> x = np.random.random((4, 4, 3)) # single RGB image
    >>> y = keras_core.ops.image.resize(x, (2, 2))
    >>> y.shape
    (2, 2, 3)

    >>> x = np.random.random((2, 3, 4, 4)) # batch of 2 RGB images
    >>> y = keras_core.ops.image.resize(x, (2, 2),
    ...     data_format="channels_first")
    >>> y.shape
    (2, 3, 2, 2)
    """

    if any_symbolic_tensors((image,)):
        return Resize(
            size,
            interpolation=interpolation,
            antialias=antialias,
            data_format=data_format,
        ).symbolic_call(image)
    return backend.image.resize(
        image,
        size,
        interpolation=interpolation,
        antialias=antialias,
        data_format=data_format,
    )


class AffineTransform(Operation):
    def __init__(
        self,
        interpolation="bilinear",
        fill_mode="constant",
        fill_value=0,
        data_format="channels_last",
    ):
        super().__init__()
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.data_format = data_format

    def call(self, image, transform):
        return backend.image.affine_transform(
            image,
            transform,
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            data_format=self.data_format,
        )

    def compute_output_spec(self, image, transform):
        if len(image.shape) not in (3, 4):
            raise ValueError(
                "Invalid image rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"image.shape={image.shape}"
            )
        if len(transform.shape) not in (1, 2):
            raise ValueError(
                "Invalid transform rank: expected rank 1 (single transform) "
                "or rank 2 (batch of transforms). Received input with shape: "
                f"transform.shape={transform.shape}"
            )
        return KerasTensor(image.shape, dtype=image.dtype)


@keras_core_export("keras_core.ops.image.affine_transform")
def affine_transform(
    image,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format="channels_last",
):
    """Applies the given transform(s) to the image(s).

    Args:
        image: Input image or batch of images. Must be 3D or 4D.
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
            Note that `"wrap"` is not supported by Torch backend.
        fill_value: Value used for points outside the boundaries of the input if
            `fill_mode="constant"`. Defaults to `0`.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, weight)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.

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
    >>> y = keras_core.ops.image.affine_transform(x, transform)
    >>> y.shape
    (2, 64, 80, 3)

    >>> x = np.random.random((64, 80, 3)) # single RGB image
    >>> transform = np.array([1.0, 0.5, -20, 0.5, 1.0, -16, 0, 0])  # shear
    >>> y = keras_core.ops.image.affine_transform(x, transform)
    >>> y.shape
    (64, 80, 3)

    >>> x = np.random.random((2, 3, 64, 80)) # batch of 2 RGB images
    >>> transform = np.array(
    ...     [
    ...         [1.5, 0, -20, 0, 1.5, -16, 0, 0],  # zoom
    ...         [1, 0, -20, 0, 1, -16, 0, 0],  # translation
    ...     ]
    ... )
    >>> y = keras_core.ops.image.affine_transform(x, transform,
    ...     data_format="channels_first")
    >>> y.shape
    (2, 3, 64, 80)
    """
    if any_symbolic_tensors((image, transform)):
        return AffineTransform(
            interpolation=interpolation,
            fill_mode=fill_mode,
            fill_value=fill_value,
            data_format=data_format,
        ).symbolic_call(image, transform)
    return backend.image.affine_transform(
        image,
        transform,
        interpolation=interpolation,
        fill_mode=fill_mode,
        fill_value=fill_value,
        data_format=data_format,
    )
