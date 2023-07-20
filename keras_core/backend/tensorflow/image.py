import tensorflow as tf

RESIZE_INTERPOLATIONS = (
    "bilinear",
    "nearest",
    "lanczos3",
    "lanczos5",
    "bicubic",
)


def resize(
    image,
    size,
    interpolation="bilinear",
    antialias=False,
    data_format="channels_last",
):
    if interpolation not in RESIZE_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{RESIZE_INTERPOLATIONS}. Received: interpolation={interpolation}"
        )
    if not len(size) == 2:
        raise ValueError(
            "Argument `size` must be a tuple of two elements "
            f"(height, width). Received: size={size}"
        )
    size = tuple(size)
    if data_format == "channels_first":
        if len(image.shape) == 4:
            image = tf.transpose(image, (0, 2, 3, 1))
        elif len(image.shape) == 3:
            image = tf.transpose(image, (1, 2, 0))
        else:
            raise ValueError(
                "Invalid input rank: expected rank 3 (single image) "
                "or rank 4 (batch of images). Received input with shape: "
                f"image.shape={image.shape}"
            )

    resized = tf.image.resize(
        image, size, method=interpolation, antialias=antialias
    )
    if data_format == "channels_first":
        if len(image.shape) == 4:
            resized = tf.transpose(resized, (0, 3, 1, 2))
        elif len(image.shape) == 3:
            resized = tf.transpose(resized, (2, 0, 1))
    return resized


AFFINE_TRANSFORM_INTERPOLATIONS = (
    "nearest",
    "bilinear",
)
AFFINE_TRANSFORM_FILL_MODES = (
    "constant",
    "nearest",
    "wrap",
    # "mirror", not supported by TF
    "reflect",
)


def affine_transform(
    image,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format="channels_last",
):
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{AFFINE_TRANSFORM_INTERPOLATIONS}. Received: "
            f"interpolation={interpolation}"
        )
    if fill_mode not in AFFINE_TRANSFORM_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{AFFINE_TRANSFORM_FILL_MODES}. Received: fill_mode={fill_mode}"
        )
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
    # unbatched case
    need_squeeze = False
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)
        need_squeeze = True
    if len(transform.shape) == 1:
        transform = tf.expand_dims(transform, axis=0)

    if data_format == "channels_first":
        image = tf.transpose(image, (0, 2, 3, 1))

    affined = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=tf.cast(transform, dtype=tf.float32),
        output_shape=tf.shape(image)[1:-1],
        fill_value=fill_value,
        interpolation=interpolation.upper(),
        fill_mode=fill_mode.upper(),
    )

    if data_format == "channels_first":
        affined = tf.transpose(affined, (0, 3, 1, 2))
    if need_squeeze:
        affined = tf.squeeze(affined, axis=0)
    return affined
