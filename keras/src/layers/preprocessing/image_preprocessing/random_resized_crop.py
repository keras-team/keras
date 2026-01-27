from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random.seed_generator import SeedGenerator


@keras_export("keras.layers.RandomResizedCrop")
class RandomResizedCrop(BaseImagePreprocessingLayer):
    """A preprocessing layer which randomly crops and resizes images.

    This layer randomly crops a portion of the image and resizes it to a target
    size. The crop area and aspect ratio are randomly sampled within specified
    ranges, matching the behavior of torchvision.transforms.RandomResizedCrop.

    During training, for each image independently:
    - Sample a crop area fraction uniformly from `scale`
    - Sample an aspect ratio uniformly in log space from `ratio`
    - Compute crop height and width using the sampled area and aspect ratio
    - Clamp crop dimensions to fit within the image bounds
    - Sample random crop offsets to position the crop within the image
    - Crop the image and resize to `(height, width)`

    At inference time, the full image is deterministically resized to
    `(height, width)` without cropping.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.

    **Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline
    (independently of which backend you're using).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format,
        or `(..., channels, height, width)`, in `"channels_first"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., target_height, target_width, channels)`,
        or `(..., channels, target_height, target_width)`,
        in `"channels_first"` format.

    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        scale: Tuple of two floats, the minimum and maximum crop area fraction.
            Defaults to `(0.08, 1.0)`.
        ratio: Tuple of two floats, the minimum and maximum aspect ratio.
            Defaults to `(3./4., 4./3.)`.
        interpolation: String, the interpolation method for resizing.
            Supports `"bilinear"` and `"nearest"`. Defaults to `"bilinear"`.
        seed: Integer. Used to create a random seed.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Example:

    ```python
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    image = x_train[0]
    cropper = keras.layers.RandomResizedCrop(height=128, width=128)
    cropped_image = cropper(image)
    print("original:", image.shape, "cropped:", cropped_image.shape)
    ```
    ```
    """

    def __init__(
        self,
        height,
        width,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
        seed=None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self.height = int(height)
        self.width = int(width)
        self.scale = scale
        self.ratio = ratio
        if interpolation not in ("nearest", "bilinear"):
            raise ValueError(
                f"Invalid interpolation method: {interpolation}. "
                "Supported methods are 'nearest' and 'bilinear'."
            )
        self.interpolation = interpolation
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.data_format = backend.standardize_data_format(data_format)

        if self.data_format == "channels_first":
            self.height_axis = -2
            self.width_axis = -1
        elif self.data_format == "channels_last":
            self.height_axis = -3
            self.width_axis = -2

        self.supports_masking = False
        self.supports_jit = False

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if isinstance(data, dict):
            input_shape = self.backend.shape(data["images"])
        else:
            input_shape = self.backend.shape(data)

        was_batched = len(input_shape) == 4

        input_height, input_width = (
            input_shape[self.height_axis],
            input_shape[self.width_axis],
        )
        if input_height is None or input_width is None:
            raise ValueError(
                "RandomResizedCrop requires the input to have a fully defined "
                f"height and width. Received: images.shape={input_shape}"
            )

        seed = seed if seed is not None else self.generator
        batch_size = input_shape[0] if was_batched else 1
        shape = (batch_size,)

        # Sample area fraction
        area_fraction = self.backend.random.uniform(
            shape=shape,
            minval=self.scale[0],
            maxval=self.scale[1],
            seed=seed,
        )
        area = area_fraction * input_height * input_width

        # Sample aspect ratio in log space
        log_ratio_min = ops.log(self.ratio[0])
        log_ratio_max = ops.log(self.ratio[1])
        log_ratio = self.backend.random.uniform(
            shape=shape,
            minval=log_ratio_min,
            maxval=log_ratio_max,
            seed=seed,
        )
        aspect_ratio = ops.exp(log_ratio)

        # Compute crop dimensions
        crop_h = ops.sqrt(area / aspect_ratio)
        crop_w = ops.sqrt(area * aspect_ratio)

        # Clamp to image bounds
        crop_h = ops.clip(crop_h, 1, input_height)
        crop_w = ops.clip(crop_w, 1, input_width)

        # Cast to int
        crop_h = ops.cast(crop_h, "int32")
        crop_w = ops.cast(crop_w, "int32")

        # Sample crop offsets
        h_start = ops.cast(
            self.backend.random.uniform(
                shape=shape,
                minval=0,
                maxval=ops.maximum(input_height - crop_h + 1, 1),
                seed=seed,
            ),
            "int32",
        )
        w_start = ops.cast(
            self.backend.random.uniform(
                shape=shape,
                minval=0,
                maxval=ops.maximum(input_width - crop_w + 1, 1),
                seed=seed,
            ),
            "int32",
        )

        transformation = {
            "h_start": h_start,
            "w_start": w_start,
            "crop_height": crop_h,
            "crop_width": crop_w,
        }

        return transformation

    def transform_images(self, images, transformation, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if training and transformation is not None:
            # Crop and resize each image
            h_start = transformation["h_start"]
            w_start = transformation["w_start"]
            crop_height = transformation["crop_height"]
            crop_width = transformation["crop_width"]

            def crop_and_resize_single_image(args):
                img, h_s, w_s, c_h, c_w = args
                cropped = self.backend.image.crop_to_bounding_box(
                    img,
                    offset_height=h_s,
                    offset_width=w_s,
                    target_height=c_h,
                    target_width=c_w,
                    data_format=self.data_format,
                )
                resized = self.backend.image.resize(
                    cropped,
                    size=(self.height, self.width),
                    interpolation=self.interpolation,
                    data_format=self.data_format,
                )
                return resized

            if len(images.shape) == 4:  # batched
                outputs = ops.map(
                    crop_and_resize_single_image,
                    (images, h_start, w_start, crop_height, crop_width),
                )
            else:  # unbatched
                outputs = crop_and_resize_single_image(
                    (images, h_start, w_start, crop_height, crop_width)
                )
        else:
            # Inference: just resize
            outputs = self.backend.image.resize(
                images,
                size=(self.height, self.width),
                interpolation=self.interpolation,
                data_format=self.data_format,
            )
        return outputs

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        return bounding_boxes

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return segmentation_masks

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[self.height_axis] = self.height
        input_shape[self.width_axis] = self.width
        return tuple(input_shape)

    def compute_output_spec(self, inputs, **kwargs):
        output_shape = self.compute_output_shape(inputs.shape)
        return backend.KerasTensor(
            output_shape, dtype=inputs.dtype, sparse=inputs.sparse
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "scale": self.scale,
                "ratio": self.ratio,
                "interpolation": self.interpolation,
                "seed": self.seed,
                "data_format": self.data_format,
            }
        )
        return config
