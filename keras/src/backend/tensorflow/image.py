import functools
import itertools
import operator

import tensorflow as tf

from keras.src.backend.tensorflow.core import convert_to_tensor

RESIZE_INTERPOLATIONS = (
    "bilinear",
    "nearest",
    "lanczos3",
    "lanczos5",
    "bicubic",
)


def rgb_to_grayscale(image, data_format="channels_last"):
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
    grayscale_image = tf.image.rgb_to_grayscale(image)
    if data_format == "channels_first":
        if len(image.shape) == 4:
            grayscale_image = tf.transpose(grayscale_image, (0, 3, 1, 2))
        elif len(image.shape) == 3:
            grayscale_image = tf.transpose(grayscale_image, (2, 0, 1))
    return tf.cast(grayscale_image, image.dtype)


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
    if interpolation not in RESIZE_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{RESIZE_INTERPOLATIONS}. Received: interpolation={interpolation}"
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
    if crop_to_aspect_ratio:
        shape = tf.shape(image)
        height, width = shape[-3], shape[-2]
        target_height, target_width = size
        crop_height = tf.cast(
            tf.cast(width * target_height, "float32") / target_width,
            "int32",
        )
        crop_height = tf.minimum(height, crop_height)
        crop_height = tf.cast(crop_height, "int32")
        crop_width = tf.cast(
            tf.cast(height * target_width, "float32") / target_height,
            "int32",
        )
        crop_width = tf.minimum(width, crop_width)
        crop_width = tf.cast(crop_width, "int32")

        crop_box_hstart = tf.cast(
            tf.cast(height - crop_height, "float32") / 2, "int32"
        )
        crop_box_wstart = tf.cast(
            tf.cast(width - crop_width, "float32") / 2, "int32"
        )
        if len(image.shape) == 4:
            image = image[
                :,
                crop_box_hstart : crop_box_hstart + crop_height,
                crop_box_wstart : crop_box_wstart + crop_width,
                :,
            ]
        else:
            image = image[
                crop_box_hstart : crop_box_hstart + crop_height,
                crop_box_wstart : crop_box_wstart + crop_width,
                :,
            ]
    elif pad_to_aspect_ratio:
        shape = tf.shape(image)
        height, width = shape[-3], shape[-2]
        target_height, target_width = size
        pad_height = tf.cast(
            tf.cast(width * target_height, "float32") / target_width,
            "int32",
        )
        pad_height = tf.maximum(height, pad_height)
        pad_height = tf.cast(pad_height, "int32")
        pad_width = tf.cast(
            tf.cast(height * target_width, "float32") / target_height,
            "int32",
        )
        pad_width = tf.maximum(width, pad_width)
        pad_width = tf.cast(pad_width, "int32")

        img_box_hstart = tf.cast(
            tf.cast(pad_height - height, "float32") / 2, "int32"
        )
        img_box_wstart = tf.cast(
            tf.cast(pad_width - width, "float32") / 2, "int32"
        )
        if len(image.shape) == 4:
            batch_size = tf.shape(image)[0]
            channels = tf.shape(image)[3]
            padded_img = tf.cond(
                img_box_hstart > 0,
                lambda: tf.concat(
                    [
                        tf.ones(
                            (batch_size, img_box_hstart, width, channels),
                            dtype=image.dtype,
                        )
                        * fill_value,
                        image,
                        tf.ones(
                            (batch_size, img_box_hstart, width, channels),
                            dtype=image.dtype,
                        )
                        * fill_value,
                    ],
                    axis=1,
                ),
                lambda: image,
            )
            padded_img = tf.cond(
                img_box_wstart > 0,
                lambda: tf.concat(
                    [
                        tf.ones(
                            (batch_size, height, img_box_wstart, channels),
                            dtype=image.dtype,
                        )
                        * fill_value,
                        padded_img,
                        tf.ones(
                            (batch_size, height, img_box_wstart, channels),
                            dtype=image.dtype,
                        )
                        * fill_value,
                    ],
                    axis=2,
                ),
                lambda: padded_img,
            )
        else:
            channels = tf.shape(image)[2]
            padded_img = tf.cond(
                img_box_hstart > 0,
                lambda: tf.concat(
                    [
                        tf.ones(
                            (img_box_hstart, width, channels),
                            dtype=image.dtype,
                        )
                        * fill_value,
                        image,
                        tf.ones(
                            (img_box_hstart, width, channels),
                            dtype=image.dtype,
                        )
                        * fill_value,
                    ],
                    axis=0,
                ),
                lambda: image,
            )
            padded_img = tf.cond(
                img_box_wstart > 0,
                lambda: tf.concat(
                    [
                        tf.ones(
                            (height, img_box_wstart, channels),
                            dtype=image.dtype,
                        )
                        * fill_value,
                        padded_img,
                        tf.ones(
                            (height, img_box_wstart, channels),
                            dtype=image.dtype,
                        )
                        * fill_value,
                    ],
                    axis=1,
                ),
                lambda: padded_img,
            )
        image = padded_img

    resized = tf.image.resize(
        image, size, method=interpolation, antialias=antialias
    )
    if data_format == "channels_first":
        if len(image.shape) == 4:
            resized = tf.transpose(resized, (0, 3, 1, 2))
        elif len(image.shape) == 3:
            resized = tf.transpose(resized, (2, 0, 1))
    return tf.cast(resized, image.dtype)


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
    affined = tf.ensure_shape(affined, image.shape)

    if data_format == "channels_first":
        affined = tf.transpose(affined, (0, 3, 1, 2))
    if need_squeeze:
        affined = tf.squeeze(affined, axis=0)
    return affined


def _mirror_index_fixer(index, size):
    s = size - 1  # Half-wavelength of triangular wave
    # Scaled, integer-valued version of the triangular wave |x - round(x)|
    return tf.abs((index + s) % (2 * s) - s)


def _reflect_index_fixer(index, size):
    return tf.math.floordiv(
        _mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2
    )


_INDEX_FIXERS = {
    "constant": lambda index, size: index,
    "nearest": lambda index, size: tf.clip_by_value(index, 0, size - 1),
    "wrap": lambda index, size: index % size,
    "mirror": _mirror_index_fixer,
    "reflect": _reflect_index_fixer,
}


def _nearest_indices_and_weights(coordinate):
    coordinate = (
        coordinate if coordinate.dtype.is_integer else tf.round(coordinate)
    )
    index = tf.cast(coordinate, tf.int32)
    weight = tf.constant(1, coordinate.dtype)
    return [(index, weight)]


def _linear_indices_and_weights(coordinate):
    lower = tf.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = tf.cast(lower, tf.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


def map_coordinates(
    input, coordinates, order, fill_mode="constant", fill_value=0.0
):
    input_arr = convert_to_tensor(input)
    coordinate_arrs = convert_to_tensor(coordinates)
    # unstack into a list of tensors for following operations
    coordinate_arrs = tf.unstack(coordinate_arrs, axis=0)
    fill_value = convert_to_tensor(tf.cast(fill_value, input_arr.dtype))

    if len(coordinates) != len(input_arr.shape):
        raise ValueError(
            "coordinates must be a sequence of length input.shape, but "
            f"{len(coordinates)} != {len(input_arr.shape)}"
        )

    index_fixer = _INDEX_FIXERS.get(fill_mode)
    if index_fixer is None:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected one of "
            f"{set(_INDEX_FIXERS.keys())}. Received: "
            f"fill_mode={fill_mode}"
        )

    def is_valid(index, size):
        if fill_mode == "constant":
            return (0 <= index) & (index < size)
        else:
            return True

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    else:
        raise NotImplementedError("map_coordinates currently requires order<=1")

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinate_arrs, input_arr.shape):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = zip(*items)
        indices = tf.transpose(tf.stack(indices))

        def fast_path():
            return tf.transpose(tf.gather_nd(input_arr, indices))

        def slow_path():
            all_valid = functools.reduce(operator.and_, validities)
            return tf.where(
                all_valid,
                tf.transpose(tf.gather_nd(input_arr, indices)),
                fill_value,
            )

        contribution = tf.cond(tf.reduce_all(validities), fast_path, slow_path)
        outputs.append(
            functools.reduce(operator.mul, weights)
            * tf.cast(contribution, weights[0].dtype)
        )
    result = functools.reduce(operator.add, outputs)
    if input_arr.dtype.is_integer:
        result = result if result.dtype.is_integer else tf.round(result)
    return tf.cast(result, input_arr.dtype)
