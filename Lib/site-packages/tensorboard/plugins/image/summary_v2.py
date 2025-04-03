# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Image summaries and TensorFlow operations to create them, V2 versions.

An image summary stores the width, height, and PNG-encoded data for zero
or more images in a rank-1 string array: `[w, h, png0, png1, ...]`.
"""


from tensorboard.compat import tf2 as tf
from tensorboard.plugins.image import metadata
from tensorboard.util import lazy_tensor_creator


def image(name, data, step=None, max_outputs=3, description=None):
    """Write an image summary.

    See also `tf.summary.scalar`, `tf.summary.SummaryWriter`.

    Writes a collection of images to the current default summary writer. Data
    appears in TensorBoard's 'Images' dashboard. Like `tf.summary.scalar` points,
    each collection of images is associated with a `step` and a `name`.  All the
    image collections with the same `name` constitute a time series of image
    collections.

    This example writes 2 random grayscale images:

    ```python
    w = tf.summary.create_file_writer('test/logs')
    with w.as_default():
      image1 = tf.random.uniform(shape=[8, 8, 1])
      image2 = tf.random.uniform(shape=[8, 8, 1])
      tf.summary.image("grayscale_noise", [image1, image2], step=0)
    ```

    To avoid clipping, data should be converted to one of the following:

    - floating point values in the range [0,1], or
    - uint8 values in the range [0,255]

    ```python
    # Convert the original dtype=int32 `Tensor` into `dtype=float64`.
    rgb_image_float = tf.constant([
      [[1000, 0, 0], [0, 500, 1000]],
    ]) / 1000
    tf.summary.image("picture", [rgb_image_float], step=0)

    # Convert original dtype=uint8 `Tensor` into proper range.
    rgb_image_uint8 = tf.constant([
      [[1, 1, 0], [0, 0, 1]],
    ], dtype=tf.uint8) * 255
    tf.summary.image("picture", [rgb_image_uint8], step=1)
    ```

    Arguments:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      data: A `Tensor` representing pixel data with shape `[k, h, w, c]`,
        where `k` is the number of images, `h` and `w` are the height and
        width of the images, and `c` is the number of channels, which
        should be 1, 2, 3, or 4 (grayscale, grayscale with alpha, RGB, RGBA).
        Any of the dimensions may be statically unknown (i.e., `None`).
        Floating point data will be clipped to the range [0,1]. Other data types
        will be clipped into an allowed range for safe casting to uint8, using
        `tf.image.convert_image_dtype`.
      step: Explicit `int64`-castable monotonic step value for this summary. If
        omitted, this defaults to `tf.summary.experimental.get_step()`, which must
        not be None.
      max_outputs: Optional `int` or rank-0 integer `Tensor`. At most this
        many images will be emitted at each step. When more than
        `max_outputs` many images are provided, the first `max_outputs` many
        images will be used and the rest silently discarded.
      description: Optional long-form description for this summary, as a
        constant `str`. Markdown is supported. Defaults to empty.

    Returns:
      True on success, or false if no summary was emitted because no default
      summary writer was available.

    Raises:
      ValueError: if a default writer exists, but no step was provided and
        `tf.summary.experimental.get_step()` is None.
    """
    summary_metadata = metadata.create_summary_metadata(
        display_name=None, description=description
    )
    # TODO(https://github.com/tensorflow/tensorboard/issues/2109): remove fallback
    summary_scope = (
        getattr(tf.summary.experimental, "summary_scope", None)
        or tf.summary.summary_scope
    )
    with summary_scope(
        name, "image_summary", values=[data, max_outputs, step]
    ) as (tag, _):
        # Defer image encoding preprocessing by passing it as a callable to write(),
        # wrapped in a LazyTensorCreator for backwards compatibility, so that we
        # only do this work when summaries are actually written.
        @lazy_tensor_creator.LazyTensorCreator
        def lazy_tensor():
            tf.debugging.assert_rank(data, 4)
            tf.debugging.assert_non_negative(max_outputs)
            images = tf.image.convert_image_dtype(data, tf.uint8, saturate=True)
            limited_images = images[:max_outputs]
            encoded_images = tf.image.encode_png(limited_images)
            image_shape = tf.shape(input=images)
            dimensions = tf.stack(
                [
                    tf.as_string(image_shape[2], name="width"),
                    tf.as_string(image_shape[1], name="height"),
                ],
                name="dimensions",
            )
            return tf.concat([dimensions, encoded_images], axis=0)

        # To ensure that image encoding logic is only executed when summaries
        # are written, we pass callable to `tensor` parameter.
        return tf.summary.write(
            tag=tag, tensor=lazy_tensor, step=step, metadata=summary_metadata
        )
